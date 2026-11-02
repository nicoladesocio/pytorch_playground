"""Dataset loader with caching support for partial downloads.

The training pipelines use a *small* random subset of ImageNet-1k rather than the
full 1.28M-image corpus: the whole point of this playground is that a training run
must finish on a normal PC (with a reasonably sized GPU) in a reasonable time.

Notes that are easy to get wrong and are therefore written down here:

* The HuggingFace id of ImageNet-1k is ``ILSVRC/imagenet-1k`` -- the bare name
  ``imagenet-1k`` is *not* a valid repo id and fails with
  ``Repository id must be 'namespace/name'``.
* ``ILSVRC/imagenet-1k`` is a **gated** dataset. Access requires an accepted
  agreement on the Hub plus a token. The token is read from the environment
  (``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``) by ``datasets`` itself; it must
  never be hardcoded into this repository.
* Streaming a gated dataset without a token raises ``DatasetNotFoundError``; we
  translate that into an actionable message instead of a bare traceback.
* Subsets are built by reading whole parquet row groups off the Hub
  (``_download_split_via_parquet``) rather than by ``streaming`` + ``shuffle(buffer)``.
  The row groups in ``ILSVRC/imagenet-1k`` are already globally shuffled, so this is
  both ~20x faster and memory-flat. The streaming path is kept as a fallback for
  datasets that are not parquet-backed.
"""

import hashlib
import itertools
import json
import os
import random
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset as HFDataset, Image
from huggingface_hub import HfFileSystem
from torch.utils.data import WeightedRandomSampler

# Canonical dataset id used by the training scripts. Keeping it in one place means
# a rename (or a swap to an ungated mirror) is a one-line change.
DEFAULT_DATASET = "ILSVRC/imagenet-1k"

# Streaming shuffle buffer. `datasets` must fill the whole buffer before it can emit
# the first sample, so the buffer size directly controls both how long the download
# stalls at "Generating train split: 0 examples" and how much RAM the process holds.
# 10_000 ImageNet frames is ~1.5 GB and was OOM-killed in the 7 GB dev container, so
# the default is deliberately modest; a few thousand is plenty for a uniform sample.
DEFAULT_SHUFFLE_BUFFER = 2048

# Repo root = .../pytorch_playground (this file lives in <root>/utils/). Defaults
# are derived from it so the scripts also work outside the /workspace container;
# the container mounts the repo *at* /workspace, so container behaviour is unchanged.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_ROOT = os.path.join(_REPO_ROOT, "Data", "datasets")


def _cache_key(dataset_name, train_samples, val_samples, target_classes, seed):
    """Stable cache directory name for one (dataset, sizes, classes, seed) tuple.

    ``dataset_name`` contains a ``/`` (namespace/name) and the class list / seed
    change what is cached, so they are folded into a short digest.
    """
    classes_tag = "auto" if target_classes is None else "fixed"
    classes_repr = ",".join(str(c) for c in sorted(target_classes)) if target_classes else "auto"
    payload = f"{dataset_name}|{train_samples}|{val_samples}|{classes_repr}|{seed}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    safe_name = dataset_name.replace("/", "__")
    return f"{safe_name}_{train_samples}t_{val_samples}v_{classes_tag}_{digest}"


def _split_shards(fs, dataset_name, split):
    """List the parquet shards backing one split of a Hub dataset."""
    prefix = f"datasets/{dataset_name}/data"
    try:
        entries = fs.ls(prefix, detail=True)
    except Exception:
        return []
    names = [e["name"] for e in entries if e["name"].endswith(".parquet")]
    return sorted(n for n in names if n.split("/")[-1].startswith(f"{split}-"))


def _row_group_plan(fs, shards, shard_order, seed, num_samples, split):
    """Deterministic ``(shard_index, row_group, num_rows)`` sequence for one split.

    Row-group order *within* a shard is shuffled from a seed derived from the shard
    index instead of from one running RNG. That makes the whole sequence a pure
    function of ``(shards, seed)`` rather than of how many reads have already
    happened, which is what lets a resumed download replay exactly the same subset
    an uninterrupted run would have produced.

    Only parquet footers are read here, so this is cheap even across hundreds of
    shards.
    """
    plan = []
    total = 0
    for shard_idx in shard_order:
        if total >= num_samples:
            break
        shard = shards[shard_idx]
        try:
            with fs.open(shard) as fh:
                meta = pq.ParquetFile(fh).metadata
                row_counts = [meta.row_group(i).num_rows
                              for i in range(meta.num_row_groups)]
        except Exception as e:
            # One bad/unreadable shard must not abort a multi-hour download.
            print(f"  [WARN] skipping shard {shard.split('/')[-1]}: "
                  f"{type(e).__name__}: {e}")
            continue
        order = list(range(len(row_counts)))
        random.Random(f"{seed}|{shard_idx}").shuffle(order)
        for rg in order:
            plan.append((shard_idx, rg, row_counts[rg]))
            total += row_counts[rg]
            if total >= num_samples:
                break
    return plan


def _resume_state(out_dir, manifest_path, plan, rows_per_part):
    """Return ``(part_paths, collected_rows, steps_done)`` for a partial download.

    Returns ``([], 0, 0)`` when nothing can be trusted, in which case the caller
    starts from scratch. Part files are only adopted if they all open *and* their
    row counts add up to the manifest's ``collected``, so a download killed
    mid-write cannot poison the cache.
    """
    if not os.path.exists(manifest_path):
        # Part files without a manifest are untrusted: we cannot tell whether the
        # last one was complete, and they may predate a plan change. Clear them so a
        # fresh download cannot silently pick up stale rows.
        for name in os.listdir(out_dir):
            if name.endswith(".parquet"):
                os.remove(os.path.join(out_dir, name))
        return [], 0, 0
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        steps_done = int(manifest["steps_done"])
        collected = int(manifest["collected"])
        expected_parts = int(manifest["parts"])
        # A different rows_per_part changes where part boundaries fall, so the
        # checkpointed part files no longer line up with the plan.
        if int(manifest["rows_per_part"]) != int(rows_per_part):
            raise ValueError("manifest was written with a different rows_per_part")
        if not (0 <= steps_done <= len(plan)) or expected_parts < 0:
            raise ValueError("manifest out of range")
        parts = []
        total = 0
        for i in range(expected_parts):
            path = os.path.join(out_dir, f"part-{i:04d}.parquet")
            total += pq.ParquetFile(path).metadata.num_rows
            parts.append(path)
        if total != collected:
            raise ValueError(f"parts hold {total} rows, manifest says {collected}")
    except Exception as e:
        print(f"  [WARN] ignoring unusable partial download in {out_dir} "
              f"({type(e).__name__}: {e}); restarting from scratch")
        for name in os.listdir(out_dir):
            if name.endswith(".parquet"):
                os.remove(os.path.join(out_dir, name))
        os.remove(manifest_path)
        return [], 0, 0

    print(f"  {manifest['split']}: resuming after {steps_done} row groups "
          f"({collected} samples, {expected_parts} part files)")
    return parts, collected, steps_done


def _download_split_via_parquet(fs, dataset_name, split, num_samples, seed, out_dir,
                                rows_per_part=2048):
    """Build a random subset by reading whole parquet row groups straight off the Hub.

    This replaces ``load_dataset(..., streaming=True).shuffle(buffer).take(n)``, which
    was unusable on ImageNet-1k. Measured in the dev container (8 CPU / 8 GB):

        streaming + shuffle(2048) :  1.5 samples/s, peak RSS 6.2 GB, OOM-killed
        sequential row-group read : 33.4 samples/s, peak RSS ~0.5 GB

    The difference is access pattern, not decoding: ``shuffle`` walks all 294 train
    shards round-robin, so it keeps many parquet readers and their compressed page
    buffers alive at once. Reading a *randomly chosen shard* and then its row groups
    sequentially is both ~22x faster and memory-flat.

    Images stay as raw encoded bytes (the parquet column is ``struct<bytes,path>``), so
    nothing is decoded here; ``ImageNetSubsetDataset`` decodes lazily inside DataLoader
    workers where the cost is parallelised.

    The download is **resumable**: progress is checkpointed to ``_manifest.json`` after
    every part file, so an interrupted 51 200-sample run (~40 min) continues where it
    stopped instead of restarting. See :func:`_row_group_plan` for why the replay is
    deterministic.

    Returns the list of part-file paths written, or ``None`` if the dataset is not
    parquet-backed (caller then falls back to streaming).
    """
    shards = _split_shards(fs, dataset_name, split)
    if not shards:
        return None

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "_manifest.json")

    rng = random.Random(seed)
    shard_order = list(range(len(shards)))
    rng.shuffle(shard_order)

    plan = _row_group_plan(fs, shards, shard_order, seed, num_samples, split)
    if not plan:
        return None

    parts, collected, steps_done = _resume_state(
        out_dir, manifest_path, plan, rows_per_part)

    buffers = []
    buffered_rows = 0

    def flush(step):
        nonlocal buffers, buffered_rows
        if not buffers:
            return
        table = pa.concat_tables(buffers)
        part_path = os.path.join(out_dir, f"part-{len(parts):04d}.parquet")
        pq.write_table(table, part_path)
        parts.append(part_path)
        buffers = []
        buffered_rows = 0
        with open(manifest_path, "w") as f:
            json.dump({"split": split, "steps_done": step, "collected": collected,
                       "parts": len(parts), "num_samples": num_samples,
                       "rows_per_part": rows_per_part, "seed": seed}, f, indent=2)
        print(f"  {split}: {collected}/{num_samples} samples "
              f"({len(parts)} part files)")

    # Group the plan by shard so each shard is opened exactly once and its row
    # groups are read sequentially -- that is what keeps memory flat. The `with`
    # guarantees the remote handle is closed even when a shard read raises.
    for shard_idx, steps in itertools.groupby(enumerate(plan), key=lambda p: p[1][0]):
        steps = list(steps)
        if steps[-1][0] < steps_done or collected >= num_samples:
            continue
        try:
            with fs.open(shards[shard_idx]) as fh:
                pf = pq.ParquetFile(fh)
                for global_step, (_idx, rg, _nrows) in steps:
                    if global_step < steps_done or collected >= num_samples:
                        continue
                    try:
                        table = pf.read_row_group(rg)
                    except Exception as e:
                        print(f"  [WARN] skipping "
                              f"{shards[shard_idx].split('/')[-1]} row group {rg}: "
                              f"{type(e).__name__}: {e}")
                        continue
                    remaining = num_samples - collected
                    if table.num_rows > remaining:
                        table = table.slice(0, remaining)
                    buffers.append(table)
                    buffered_rows += table.num_rows
                    collected += table.num_rows
                    if buffered_rows >= rows_per_part:
                        flush(global_step + 1)
        except Exception as e:
            print(f"  [WARN] skipping shard {shards[shard_idx].split('/')[-1]}: "
                  f"{type(e).__name__}: {e}")

    flush(len(plan))
    if collected < num_samples:
        print(f"  [WARN] only {collected}/{num_samples} samples could be read from "
              f"{len(shards)} shards of split '{split}'.")
    if not parts:
        return None
    return parts


def _raise_load_help(dataset_name, exc):
    """Raise a RuntimeError with actionable guidance for the common failure modes."""
    text = str(exc)
    lowered = text.lower()
    if "gated" in lowered or "authenticated" in lowered:
        detail = (
            f"'{dataset_name}' is a gated dataset on the HuggingFace Hub. To unlock it:\n"
            f"  1. Open https://huggingface.co/datasets/{dataset_name} while logged in and "
            "accept the access conditions (ImageNet-1k requires the ILSVRC terms).\n"
            "  2. Create a token at https://huggingface.co/settings/tokens and export it:\n"
            "       export HF_TOKEN=hf_...          # Linux/macOS\n"
            "       $env:HF_TOKEN='hf_...'          # PowerShell\n"
            "  3. Re-run the same command. The token is only ever read from the environment."
        )
    elif "namespace/name" in text:
        detail = (
            f"'{dataset_name}' is not a valid Hub repo id -- it must be 'namespace/name'. "
            f"Did you mean '{DEFAULT_DATASET}'?"
        )
    else:
        detail = f"{type(exc).__name__}: {text}"
    raise RuntimeError(f"Could not load dataset '{dataset_name}'.\n{detail}") from exc


def _as_undecoded_images(dataset: HFDataset) -> HFDataset:
    """Store ``Image`` columns as raw encoded bytes instead of decoded PIL images.

    Only pays for a real change when a column is an ``Image`` feature that would
    decode. The parquet path stores raw JPEG bytes (arrow type
    ``struct<bytes,path>``), but ``HFDataset.from_parquet`` restores
    ``Image(decode=True)`` from the metadata embedded in the file, so this still has
    to flip the feature -- it just must not do it the expensive way.

    Measured on the 4096-sample cached subset in the dev container:

        ``dataset.cast(features)``            126.14 s   (~26 min for 51 200 samples)
        ``dataset.cast_column(name, Image())``  0.17 s
        ``dataset._info.features = ...``        0.12 s

    All three yield ``Image(decode=False)`` and return the identical
    ``{'bytes': ..., 'path': ...}`` dict at runtime. ``cast`` re-runs the whole arrow
    table through the decoder/re-encoder even though the physical type is unchanged,
    so ``cast_column`` is used for the single-image case (the only case this dataset
    has) and ``cast`` is kept as the multi-column fallback.
    """
    features = dataset.features
    image_names = [
        name for name, feat in features.items()
        if isinstance(feat, Image) and getattr(feat, "decode", False)
    ]
    if not image_names:
        return dataset
    if len(image_names) == 1:
        return dataset.cast_column(image_names[0], Image(decode=False))
    new_features = features.copy()
    for name in image_names:
        new_features[name] = Image(decode=False)
    return dataset.cast(new_features)


def _download_via_streaming(
    dataset_name: str,
    train_split: str,
    val_split: str,
    train_samples: int,
    val_samples: int,
    seed: int,
    shuffle_buffer_size: int,
):
    """Fallback path for datasets that are not parquet-backed on the Hub.

    Kept so non-parquet datasets still work. It is roughly 20x slower and far more
    memory hungry than the row-group path, and it *does* need the shuffle buffer,
    because streaming yields records in shard order.
    """
    try:
        dataset = load_dataset(dataset_name, streaming=True)
    except Exception as e:
        _raise_load_help(dataset_name, e)

    def _undecoded(split_name):
        split = dataset[split_name]
        features = split.features.copy()
        image_fields = [n for n, f in features.items() if isinstance(f, Image)]
        if len(image_fields) == 1:
            return split.cast_column(image_fields[0], Image(decode=False))
        for field_name in image_fields:
            features[field_name] = Image(decode=False)
        return split.cast(features)

    train_split_stream = _undecoded(train_split)
    val_split_stream = _undecoded(val_split)

    def sample_generator(split_stream, num_samples):
        shuffled = split_stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        for i, sample in enumerate(shuffled.take(num_samples)):
            yield sample
            if (i + 1) % 1024 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples from split...")

    print("Processing train subset...")
    train_dataset = HFDataset.from_generator(
        sample_generator,
        features=train_split_stream.features,
        gen_kwargs={"split_stream": train_split_stream, "num_samples": train_samples},
    )
    print("Processing val subset...")
    validation_dataset = HFDataset.from_generator(
        sample_generator,
        features=val_split_stream.features,
        gen_kwargs={"split_stream": val_split_stream, "num_samples": val_samples},
    )
    return train_dataset, validation_dataset


def load_dataset_subset(
    dataset_name=DEFAULT_DATASET,
    cache_root=DEFAULT_CACHE_ROOT,
    train_split="train",
    val_split="validation",
    train_samples=51200,
    val_samples=2048,
    target_classes=None,
    seed=42,
    shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER,
):
    """Load (and cache) a small random subset of a HuggingFace image dataset.

    Returns ``(train_dataset, validation_dataset, target_classes)`` where
    ``target_classes`` is the sorted list of *original* class ids the cached subset
    covers, or ``None`` when the caller wants the downstream sampler to decide.

    Subsets are drawn by sampling whole parquet row groups off the Hub. In
    ``ILSVRC/imagenet-1k`` the row groups are already globally shuffled -- a single
    872-row group spans 575 distinct labels across the full 0..999 range -- so
    reading whole groups gives a class cross-section for free, and is both ~20x
    faster and memory-flat compared with ``streaming`` + ``shuffle(buffer)``.
    ``shuffle_buffer_size`` is only used by the streaming fallback.
    """
    cache_dir = os.path.join(
        cache_root, _cache_key(dataset_name, train_samples, val_samples, target_classes, seed)
    )
    train_cache = os.path.join(cache_dir, train_split)
    val_cache = os.path.join(cache_dir, val_split)
    metadata_file = os.path.join(cache_dir, "metadata.json")

    expected_metadata = {
        "dataset_name": dataset_name,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "target_classes": sorted(target_classes) if target_classes is not None else None,
        "seed": seed,
        "train_split": train_split,
        "val_split": val_split,
        "shuffle_buffer_size": shuffle_buffer_size,
    }

    # Only compare the fields that actually determine the cached bytes.
    # ``shuffle_buffer_size`` is recorded for provenance but must not invalidate the
    # cache: the parquet path ignores it (row groups are already globally shuffled), so
    # passing a different --shuffle_buffer used to force a full re-download of a
    # byte-identical subset. It is also absent from _cache_key, so the two checks
    # disagreed about what "the same subset" means.
    #
    # The split names stay in the comparison (they change the contents), even though
    # _cache_key does not include them. _cache_key is deliberately left alone: adding
    # fields to it would invalidate every existing cache, including multi-hour
    # downloads already in flight.
    identity_keys = tuple(k for k in expected_metadata if k != "shuffle_buffer_size")

    train_dataset = None
    validation_dataset = None

    if all(os.path.exists(p) for p in (metadata_file, train_cache, val_cache)):
        try:
            with open(metadata_file) as f:
                cached = json.load(f)
            if all(cached.get(k) == v for k, v in expected_metadata.items()
                   if k in identity_keys):
                print(f"Loading cached subset of {dataset_name} from {cache_dir}")
                train_dataset = HFDataset.load_from_disk(train_cache)
                validation_dataset = HFDataset.load_from_disk(val_cache)
            else:
                print("Cache metadata mismatch, re-downloading...")
        except Exception as e:  # a corrupt cache must never hard-fail a run
            print(f"Error reading cache metadata ({e}), re-downloading...")

    if train_dataset is None:
        print(f"Downloading {dataset_name} subset "
              f"({train_samples} train, {val_samples} val samples)...")
        print("This happens only once; the result is cached under the cache root.")

        os.makedirs(cache_dir, exist_ok=True)
        fs = HfFileSystem()

        # Preferred path: read whole parquet row groups off the Hub. See
        # _download_split_via_parquet for the measurements that motivated it.
        train_parts = _download_split_via_parquet(
            fs, dataset_name, train_split, train_samples, seed,
            os.path.join(cache_dir, "_train_parts"))
        val_parts = _download_split_via_parquet(
            fs, dataset_name, val_split, val_samples, seed,
            os.path.join(cache_dir, "_val_parts"))

        if train_parts and val_parts:
            train_dataset = HFDataset.from_parquet(train_parts)
            validation_dataset = HFDataset.from_parquet(val_parts)
            # Store the images undecoded. Decoding here would cost a decode +
            # re-encode round trip on every sample; ImageNetSubsetDataset decodes
            # lazily inside DataLoader workers instead.
            train_dataset = _as_undecoded_images(train_dataset)
            validation_dataset = _as_undecoded_images(validation_dataset)
            train_dataset.save_to_disk(train_cache)
            validation_dataset.save_to_disk(val_cache)
            shutil.rmtree(os.path.join(cache_dir, "_train_parts"), ignore_errors=True)
            shutil.rmtree(os.path.join(cache_dir, "_val_parts"), ignore_errors=True)
        else:
            print("Dataset is not parquet-backed; falling back to streaming.")
            train_dataset, validation_dataset = _download_via_streaming(
                dataset_name, train_split, val_split, train_samples, val_samples,
                seed, shuffle_buffer_size)
            train_dataset.save_to_disk(train_cache)
            validation_dataset.save_to_disk(val_cache)

        with open(metadata_file, "w") as f:
            json.dump(expected_metadata, f, indent=2)
        print("Cached for future runs!")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    return train_dataset, validation_dataset, expected_metadata["target_classes"]


def label_distribution(dataset, label_key="label"):
    """Return ``{class_id: count}`` for a cached split (local only, no network)."""
    counts = {}
    for label in dataset[label_key]:
        counts[label] = counts.get(label, 0) + 1
    return counts


def choose_target_classes(train_dataset, num_classes, seed=42, min_count=1, label_key="label",
                          val_dataset=None, min_val_count=1):
    """Pick a deterministic, *usable* class subset from the cached train split.

    The previous implementation did ``random.sample(dataset.unique("label"), k)``,
    which was unseeded (so runs were not reproducible) and blind to how many
    examples each class actually had: a class with a single cached image could be
    selected while a class with 200 was skipped, which cripples convergence.

    Selection rule: keep classes with at least ``min_count`` cached train examples
    (and, when ``val_dataset`` is given, at least ``min_val_count`` validation
    examples -- otherwise accuracy is measured on a handful of images for most
    classes), rank by count descending then class id ascending so the ranking is
    total, and break ties among equally-populated classes with a seeded RNG.
    """
    counts = label_distribution(train_dataset, label_key=label_key)
    if val_dataset is not None:
        val_counts = label_distribution(val_dataset, label_key=label_key)
        eligible = [(cls, n) for cls, n in counts.items()
                    if n >= min_count and val_counts.get(cls, 0) >= min_val_count]
    else:
        eligible = [(cls, n) for cls, n in counts.items() if n >= min_count]
    if not eligible:
        raise ValueError(
            f"No class in the cached train split has >= {min_count} samples"
            + (f" and >= {min_val_count} validation samples" if val_dataset is not None else "")
            + "; increase train_samples/val_samples or lower min_count."
        )

    rng = random.Random(seed)
    # Shuffle first so ranking by (-count, id) resolves equal counts through the
    # seeded permutation rather than through dict insertion order.
    rng.shuffle(eligible)
    eligible.sort(key=lambda cn: (-cn[1], cn[0]))

    if len(eligible) < num_classes:
        print(
            f"[WARN] Only {len(eligible)} classes have >= {min_count} cached samples "
            f"(requested {num_classes}); training on {len(eligible)} classes. "
            "Increase train_samples if you need more classes."
        )
    chosen = sorted(cls for cls, _ in eligible[:num_classes])
    per_class = [counts[c] for c in chosen]
    print(
        f"Selected {len(chosen)} classes; per-class cached counts: "
        f"min={min(per_class)}, median={sorted(per_class)[len(per_class) // 2]}, max={max(per_class)}"
    )
    return chosen


def create_subset_sampler(
    dataset,
    max_num_classes,
    target_classes=None,
    seed=42,
    num_samples=None,
    label_key="label",
    val_dataset=None,
):
    """Weighted sampler that restricts an epoch to ``target_classes``.

    Returns ``(sampler, target_classes)``.

    ``num_samples`` overrides the epoch length. By default it is the number of
    in-class examples, i.e. exactly one pass over them with no repeats. If the
    caller asks for more samples than exist in-class we fall back to sampling with
    replacement rather than silently returning a near-empty epoch.

    ``val_dataset`` is forwarded to :func:`choose_target_classes` so that classes
    without validation images are never selected.
    """
    if target_classes is None:
        target_classes = choose_target_classes(
            dataset, max_num_classes, seed=seed, label_key=label_key,
            val_dataset=val_dataset,
        )
    target_classes = list(target_classes)
    target_set = set(target_classes)

    weights = [1.0 if x in target_set else 0.0 for x in dataset[label_key]]
    num_in_class = int(sum(weights))
    if num_in_class == 0:
        raise ValueError(
            f"None of the {len(dataset)} samples in this split belong to "
            f"target_classes={sorted(target_set)}. Check that the dataset was cached correctly."
        )

    if num_samples is None:
        num_samples, replacement = num_in_class, False
    else:
        num_samples = int(num_samples)
        replacement = num_samples > num_in_class

    return (
        WeightedRandomSampler(
            weights=weights, num_samples=num_samples, replacement=replacement
        ),
        target_classes,
    )


def create_eval_sampler(dataset, target_classes, label_key="label"):
    """Deterministic full-coverage sampler for validation.

    Validation must see every in-class sample exactly once. Reusing
    ``create_subset_sampler`` there (as the original code did) together with
    ``drop_last=True`` in the DataLoader could silently drop the entire validation
    set when it was smaller than one batch.
    """
    target_set = set(target_classes)
    weights = [1.0 if x in target_set else 0.0 for x in dataset[label_key]]
    num_in_class = int(sum(weights))
    if num_in_class == 0:
        raise ValueError("Validation split contains none of the target classes.")
    return WeightedRandomSampler(
        weights=weights, num_samples=num_in_class, replacement=False
    )
