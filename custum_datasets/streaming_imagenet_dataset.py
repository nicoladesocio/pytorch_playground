"""Streaming ImageNet-subset dataset with a growing local disk cache.

Unlike utils.datasets_utils.load_dataset_subset (which fully downloads and
materializes `train_samples` images to disk *before* returning, blocking
training start), this dataset starts yielding samples immediately:

- Anything already cached locally (from a previous run) is read straight off
  disk - no network involved.
- Anything not yet cached is pulled from the live HuggingFace stream, and
  handed to the training loop right away while a background thread writes it
  to the local cache for next time. Downloading and training therefore
  overlap instead of the old "download everything, then train" sequence.

Trade-offs (accepted by design, see AlexNet/train.py, GoogLeNet/train.py):
- No `__len__`/fixed epoch content: `num_samples` bounds how many examples
  each pass over the dataset yields, but which images end up in that budget
  can differ between epochs/restarts (dedup is best-effort, not exact).
- Restarting mid-epoch/mid-training is fine: only the chosen `target_classes`
  need to match (restored from the checkpoint), not the exact images seen.
"""
import hashlib
import io
import os
import random

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info

from custum_datasets.imagenet_subset_dataset import build_imagenet_transform
from custum_datasets.local_image_cache_dataset import LocalImageCacheDataset

SHUFFLE_BUFFER_SIZE = 10_000


def pick_target_classes(dataset_name, num_classes, split="train", seed=None):
    """Pick `num_classes` random class ids from the dataset's label space.

    Only reads dataset *metadata* (the ClassLabel feature), not any actual
    examples, so this is effectively instant and involves no bulk download.
    """
    probe_split = load_dataset(dataset_name, split=split, streaming=True)
    num_available_classes = probe_split.features["label"].num_classes
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_available_classes), min(num_classes, num_available_classes)))


def _cache_dir_for(cache_root, dataset_name, target_classes, split):
    classes_signature = ",".join(str(c) for c in sorted(target_classes))
    class_filter_key = "classes_" + hashlib.md5(classes_signature.encode()).hexdigest()[:10]
    return os.path.join(cache_root, dataset_name, class_filter_key, split)


class StreamingImageNetSubsetDataset(IterableDataset):
    """Yields up to `num_samples` (image_tensor, label_idx) pairs per epoch,
    drawing first from a local disk cache, then from a live HF stream
    (caching newly streamed images in the background for next time).
    """

    def __init__(
        self,
        dataset_name,
        target_classes,
        num_samples,
        cache_root="/workspace/Data/datasets/",
        split="train",
        image_size=227,
        is_training=True,
        seed=42,
    ):
        self.dataset_name = dataset_name
        self.target_classes = list(target_classes)
        self.target_classes_set = set(target_classes)
        self.num_samples = num_samples
        self.split = split
        self.seed = seed
        self.cache_dir = _cache_dir_for(cache_root, dataset_name, self.target_classes, split)
        self.transform = build_imagenet_transform(image_size=image_size, is_training=is_training)

    def __len__(self):
        # Best-effort epoch size for progress bars / Lightning; the actual
        # count can be slightly lower if the live stream runs dry.
        return self.num_samples

    def _iter_live_stream(self, worker_id, num_workers, epoch_seed):
        hf_stream = load_dataset(self.dataset_name, split=self.split, streaming=True)
        hf_stream = hf_stream.filter(lambda ex: ex["label"] in self.target_classes_set)
        hf_stream = hf_stream.shuffle(seed=epoch_seed, buffer_size=SHUFFLE_BUFFER_SIZE)
        if num_workers > 1:
            try:
                hf_stream = split_dataset_by_node(hf_stream, rank=worker_id, world_size=num_workers)
            except Exception:
                # Fall back to unsharded streaming: workers may see overlapping
                # examples, which is acceptable (see module docstring).
                pass
        for item in hf_stream:
            image = item["image"].convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            yield buf.getvalue(), item["label"]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        # Vary the shuffle seed per-epoch (via a fresh RNG state each __iter__
        # call) so re-runs/resumes don't replay an identical order forever.
        epoch_seed = self.seed + random.Random().randrange(1_000_000)

        # Fresh instance per __iter__ call: it re-scans the cache directory
        # (picking up files written since last epoch/by other workers) and
        # starts its own background writer thread scoped to this worker
        # process.
        cache_dataset = LocalImageCacheDataset(self.cache_dir, self.target_classes, self.transform)

        count = 0
        target_count = max(1, self.num_samples // num_workers)

        # Shard-grouped order: whole shards go to this worker, shard order
        # and within-shard order are shuffled, but different shards' rows
        # are never interleaved - keeps LocalImageCacheDataset's shard LRU
        # cache effective instead of thrashing under a fully global shuffle.
        cached_indices = cache_dataset.shuffled_indices(epoch_seed, worker_id, num_workers)
        for idx in cached_indices:
            if count >= target_count:
                break
            yield cache_dataset[idx]
            count += 1

        try:
            if count < target_count:
                for image_bytes, label in self._iter_live_stream(worker_id, num_workers, epoch_seed):
                    if count >= target_count:
                        break
                    cache_dataset.add(label, image_bytes)
                    yield cache_dataset.decode(image_bytes, label)
                    count += 1
        finally:
            # Force any partially-filled (below flush_every) per-label
            # buffers to disk now, since this worker process (and its
            # background writer thread) may be torn down once this epoch
            # ends - otherwise those buffered-but-not-yet-flushed images
            # would be silently lost.
            cache_dataset.flush()
