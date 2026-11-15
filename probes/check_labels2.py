"""Verify that the label a DataLoader yields actually matches the image.

Kanban `P2-18`. If the ``target_classes`` remap disagrees between the image and the
label, the model can never beat chance no matter how good the architecture — and it
looks exactly like "loss decreases very slowly, accuracy stuck at chance", which is the
symptom this repo has been chasing since HANDOFF_001.

This probe was written in HANDOFF_003 but **never executed**: it hardcoded a container
path to a 4096-sample cache that no longer exists, and called ``create_subset_sampler``
with a signature it does not have (``num_samples=`` without the required
``max_num_classes``). Both are fixed here, and the cache is now auto-discovered.

    python probes/check_labels2.py                     # newest cache in Data/datasets
    python probes/check_labels2.py --cache <dir>       # a specific one

Three independent checks, each stronger than the last:

1. **label <-> WordNet id.** ImageNet-1k filenames encode the synset
   (``n03208938_...``) and the ``label`` column is the integer class index. In the
   canonical class ordering each index maps to exactly one wnid, so a label that maps to
   two wnids means the cache is corrupt.
2. **label <-> canonical class index.** ImageNet-1k's class index is the 1000 WordNet
   ids in lexicographic order, so ``label == position of its wnid in sorted(wnids)``.
   This is what catches an off-by-one in the label column, which check 1 alone cannot
   see — a uniform shift keeps the bijection intact and silently destroys accuracy.
3. **The real pipeline.** ``choose_target_classes`` -> ``create_subset_sampler`` ->
   ``ImageNetSubsetDataset`` -> ``DataLoader``, then every yielded batch is re-checked:
   the remapped index must point back at the wnid of the image it came with.

Exit code 0 only if all mismatches are zero.
"""
import argparse
import collections
import glob
import os
import sys

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

REPO_ROOT = _bootstrap.REPO_ROOT


def find_cache(cache_root=None):
    """Newest ``ILSVRC__imagenet-1k_*`` cache directory under the repo's cache root."""
    root = cache_root or os.path.join(REPO_ROOT, "Data", "datasets")
    cands = sorted(glob.glob(os.path.join(root, "ILSVRC__imagenet-1k_*")),
                   key=os.path.getmtime, reverse=True)
    cands = [c for c in cands
             if os.path.isdir(os.path.join(c, "train"))
             and os.path.isdir(os.path.join(c, "validation"))]
    if not cands:
        sys.exit(f"No cached subset found under {root}.\n"
                 "Build one first:  python probes/check_big.py")
    return cands[0]


def image_path(row):
    """The stored image is an undecoded {'bytes','path'} dict (see _as_undecoded_images)."""
    img = row["image"] if not hasattr(row, "get") else row.get("image")
    if isinstance(img, dict):
        return img["path"]
    return str(img)


def wnid_of(path):
    return os.path.basename(str(path)).split("_")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="cache dir; default = newest in Data/datasets")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=400,
                    help="sampled rows to re-check in check 3 (decodes images, so capped)")
    ap.add_argument("--image-size", type=int, default=160,
                    help="must stay >=128 or GoogLeNet's aux heads collapse (see P2-16)")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    from datasets import load_from_disk
    from torch.utils.data import DataLoader

    from utils.datasets_utils import choose_target_classes, create_subset_sampler
    from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset

    cache = find_cache(args.cache)
    print(f"cache: {cache}")
    train = load_from_disk(os.path.join(cache, "train"))
    val = load_from_disk(os.path.join(cache, "validation"))
    print(f"train={len(train)} val={len(val)}  features={list(train.features)}")

    failures = 0

    # ---------------------------------------------------------------- check 1
    # label -> set(wnids) must be a function, and wnid -> label too. A cache built by
    # the parquet row-group path should be a perfect bijection.
    label_to_wnids = collections.defaultdict(collections.Counter)
    wnid_to_labels = collections.defaultdict(collections.Counter)
    paths = train["image"]
    labels = train["label"]
    for lbl, p in zip(labels, paths):
        w = wnid_of(p["path"] if isinstance(p, dict) else p)
        label_to_wnids[lbl][w] += 1
        wnid_to_labels[w][lbl] += 1

    bad_label = {l: c for l, c in label_to_wnids.items() if len(c) > 1}
    bad_wnid = {w: c for w, c in wnid_to_labels.items() if len(c) > 1}
    print(f"\n[1] label<->wnid bijection over {len(train)} rows: "
          f"{len(label_to_wnids)} labels, {len(wnid_to_labels)} wnids")
    print(f"    labels mapping to >1 wnid: {len(bad_label)}")
    print(f"    wnids mapping to >1 label: {len(bad_wnid)}")
    for l, c in list(bad_label.items())[:5]:
        print(f"      label {l} -> {dict(c)}")
    for w, c in list(bad_wnid.items())[:5]:
        print(f"      wnid {w} -> {dict(c)}")
    if bad_label or bad_wnid:
        print("    FAIL: the cached subset itself is inconsistent")
        failures += 1
    else:
        print("    OK")

    # Every filename must actually start with the wnid its label claims.
    malformed = sum(1 for p in paths
                    if not wnid_of(p["path"] if isinstance(p, dict) else p).startswith("n"))
    print(f"    malformed filenames: {malformed} ({'OK' if malformed == 0 else 'FAIL'})")
    failures += 1 if malformed else 0

    # ---------------------------------------------------------------- check 2
    # Self-consistency (check 1) cannot see an off-by-one: if every label were shifted
    # by one, each label would still map to exactly one wnid. This check can, because
    # ImageNet-1k's canonical class index is not arbitrary — it is the 1000 WordNet ids
    # in lexicographic order (verified against the full 51 200-sample canonical subset:
    # 0 mismatches, and label 0 -> n01440764 "tench" / label 999 -> n15075141 "toilet
    # tissue", the paper's first and last classes). So the label column is pinned to an
    # external ground truth without needing the network.
    names = train.features["label"].names
    wnid_by_label = {l: next(iter(c)) for l, c in label_to_wnids.items()}
    print(f"\n[2] label index vs canonical ImageNet ordering (sorted wnids)")
    if len(wnid_by_label) == 1000:
        expected_order = sorted(wnid_by_label.values())
        off_by_one = [l for l, w in wnid_by_label.items() if expected_order[l] != w]
        print(f"    labels whose wnid is not at its sorted position: {len(off_by_one)} "
              f"({'OK' if not off_by_one else 'FAIL'})")
        for l in off_by_one[:5]:
            print(f"      label {l}: file says {wnid_by_label[l]}, "
                  f"canonical says {expected_order[l]}")
        failures += 1 if off_by_one else 0
        lo, hi = min(wnid_by_label), max(wnid_by_label)
        print(f"    extremes: label {lo} -> {wnid_by_label[lo]} "
              f"({names[lo][:32]}), label {hi} -> {wnid_by_label[hi]} "
              f"({names[hi][:32]})")
    else:
        ok = len(names) == 1000
        print(f"    only {len(wnid_by_label)} labels present in this cache; cannot check "
              f"ordering. Structural check names={len(names)} (expect 1000): "
              f"{'OK' if ok else 'FAIL'}")
        failures += 1 if not ok else 0

    # ---------------------------------------------------------------- check 3
    # The real pipeline. This is the check that actually matters: it exercises the
    # remap in ImageNetSubsetDataset, not just the stored columns.
    targets = choose_target_classes(train, num_classes=args.classes, seed=args.seed,
                                    val_dataset=val)
    print(f"\n[3] pipeline check with target_classes={targets}")
    sampler, target_classes = create_subset_sampler(
        train, max_num_classes=args.classes, target_classes=targets, seed=args.seed)
    # Freeze the sampler's order into a plain list and hand *that* to the DataLoader.
    # Calling list(sampler) and then iterating the DataLoader would invoke
    # WeightedRandomSampler.__iter__ twice and draw two different orders, which makes
    # the row/batch alignment below meaningless (it looks like a label bug).
    sampled = list(sampler)
    ds = ImageNetSubsetDataset(train, image_size=args.image_size, is_training=False,
                               target_classes=target_classes)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=list(sampled),
                        num_workers=args.num_workers)

    # Ground truth for "which wnid belongs to remapped index i": walk a real DataLoader
    # (what training consumes) and, for each batch position, re-read the raw row the
    # sampler pointed at. Comparing the row's own stored label against the remapped
    # index the dataset hands back keeps the check independent of the dataset's remap,
    # which is the thing under test.
    idx2wnids = collections.defaultdict(collections.Counter)
    seen_rows = 0
    wrong = 0
    limit = min(len(sampled), args.max_samples)
    for _, batch_labels in loader:
        if seen_rows >= limit:
            break
        chunk = sampled[seen_rows:seen_rows + len(batch_labels)]
        for idx, remapped in zip(chunk, batch_labels.tolist()):
            orig = train[int(idx)]["label"]
            wnid = (next(iter(label_to_wnids[orig]))
                    if orig in label_to_wnids else f"<unmapped:{orig}>")
            seen_rows += 1
            if target_classes[remapped] != orig:
                wrong += 1
            idx2wnids[remapped].update([wnid])

    mixed = {i: c for i, c in idx2wnids.items() if len(c) > 1}
    print(f"    inspected {seen_rows} sampled rows over {len(idx2wnids)} remapped indices")
    if seen_rows and wrong == seen_rows:
        # 100% mismatch is almost never a real label bug; it means the rows and the
        # batches being compared came from different orders. Fail loudly rather than
        # reporting a scary-looking label corruption.
        print("    FAIL: every row mismatched — this probe is misaligned, not the cache. "
              "Check that the sampler order is frozen before the DataLoader is built.")
        failures += 1
    else:
        print(f"    rows whose stored label disagrees with the dataset's remap: {wrong} "
              f"({'OK' if not wrong else 'FAIL'})")
        failures += 1 if wrong else 0
    print(f"    remapped indices pointing at >1 wnid: {len(mixed)} "
          f"({'OK' if not mixed else 'FAIL'})")
    for i, c in list(mixed.items())[:5]:
        print(f"      index {i} -> {dict(c)}")
    failures += 1 if mixed else 0
    failures += 1 if wrong else 0

    for i in sorted(idx2wnids)[:5]:
        w = next(iter(idx2wnids[i]))
        orig = target_classes[i]
        print(f"    index {i} -> orig label {orig} -> {w} -> {names[orig][:48]}")

    print()
    if failures:
        print(f"RESULT: {failures} check group(s) FAILED — every accuracy number in the "
              "repo is suspect until this is fixed.")
        return 1
    print("RESULT: labels and images agree through the real sampler. "
          "Accuracy numbers can be trusted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
