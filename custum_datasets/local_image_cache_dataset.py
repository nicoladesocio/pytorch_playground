"""Standalone, map-style dataset over a growing on-disk image cache.

Like ImagesFolderDataset, but organized for lazy, shard-at-a-time loading
instead of keeping the whole cache in RAM:

- Per label, images are stored in fixed-size Arrow (Feather/IPC) data shards
  under `cache_dir/<label>/shard_<uuid>.arrow` (up to `flush_every` images
  each). Incoming images are buffered in RAM and only flushed to disk once a
  shard's worth has accumulated (or `.flush()` is called).
- Per label, a single small `arrow_hashes.arrow` file maps
  `hash -> (shard_file, row_idx)`. This is loaded once at startup and gives
  us, cheaply: `__len__` (row count, no need to touch data shards) and an
  in-memory dedup check for `.add()` (skip images already cached).
- `__getitem__` looks up which shard a row lives in and lazily loads just
  that shard (kept in a small LRU cache of `max_cached_shards` shards) -
  the full cache is never loaded into memory at once.

Important: because only ~1-2 shards are ever resident in memory, callers
should iterate in *shard-grouped* order (see `shuffled_indices()`) rather
than fully-randomizing the global index space - a global shuffle would
thrash the shard cache, reloading a shard from disk on nearly every access.

Also owns the *write* side: `.add(label, image_bytes)` lets a caller (e.g.
StreamingImageNetSubsetDataset) persist newly-seen images into the cache.
Writes are handed off to a background thread so callers never block on disk
I/O. Call `.flush()` when done adding (e.g. end of an epoch) to force any
partially-filled, below-threshold buffer to disk - otherwise it's lost when
the process exits.

This class has no knowledge of HuggingFace/streaming at all - it only knows
how to read and write the local cache directory. It can be used standalone
(e.g. to train purely off a previously warmed cache, with no network).
"""
import hashlib
import io
import os
import queue
import random
import threading
import uuid

import pyarrow as pa
from pyarrow import feather
from PIL import Image
from torch.utils.data import Dataset

HASHES_FILENAME = "arrow_hashes.arrow"


class LocalImageCacheDataset(Dataset):
    """Read (and write) images cached as Arrow shards under
    `cache_dir/<label>/*.arrow`, with lazy per-shard loading."""

    def __init__(self, cache_dir, target_classes, transform, flush_every=256, max_cached_shards=2):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.target_classes = list(target_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.target_classes)}
        self.transform = transform
        self.flush_every = flush_every
        self.max_cached_shards = max_cached_shards

        # LRU cache of loaded shard tables, keyed by (label, shard_file).
        self._shard_cache = {}
        self._shard_cache_order = []

        # entries[i] = (label, shard_file, row_idx); built from each label's
        # arrow_hashes.arrow file (also populates self._known_hashes).
        self._known_hashes = {}
        self.entries = self._scan()

        # Per-label in-RAM buffers of not-yet-flushed (image_bytes, hash)
        # pairs. Only ever touched from the writer thread, except during
        # flush() which is only safe once no more add() calls are in flight.
        self._buffers = {}
        self._pending_hashes = {}
        self._write_queue = queue.Queue(maxsize=256)
        self._write_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._write_thread.start()

    def _scan(self):
        """Load each label's arrow_hashes.arrow file (small: hash + shard
        location per cached image) to build the (label, shard_file, row_idx)
        index and the in-memory dedup hash sets - without touching any of
        the (much larger) data shards."""
        entries = []
        for label in self.target_classes:
            self._known_hashes[label] = set()
            hashes_path = os.path.join(self.cache_dir, str(label), HASHES_FILENAME)
            if not os.path.isfile(hashes_path):
                continue
            try:
                table = feather.read_table(hashes_path)
            except (pa.ArrowInvalid, OSError):
                continue
            hashes = table.column("hash").to_pylist()
            shard_files = table.column("shard_file").to_pylist()
            row_idxs = table.column("row_idx").to_pylist()
            self._known_hashes[label] = set(hashes)
            for shard_file, row_idx in zip(shard_files, row_idxs):
                entries.append((label, shard_file, row_idx))
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        label, shard_file, row_idx = self.entries[idx]
        image_bytes = self._read_from_shard(label, shard_file, row_idx)
        return self.decode(image_bytes, label)

    def _read_from_shard(self, label, shard_file, row_idx):
        key = (label, shard_file)
        table = self._shard_cache.get(key)
        if table is None:
            shard_path = os.path.join(self.cache_dir, str(label), shard_file)
            table = feather.read_table(shard_path)
            self._shard_cache[key] = table
            self._shard_cache_order.append(key)
            if len(self._shard_cache_order) > self.max_cached_shards:
                oldest = self._shard_cache_order.pop(0)
                self._shard_cache.pop(oldest, None)
        return table.column("image")[row_idx].as_py()

    def shuffled_indices(self, seed, worker_id=0, num_workers=1):
        """Return an index ordering suitable for sequential access: whole
        shards are assigned to a single worker (never split across workers),
        shard visit order is shuffled, and rows within each shard are
        shuffled - but two different shards' rows are never interleaved.
        This keeps the shard LRU cache effective (at most 1-2 shards loaded
        at once) instead of thrashing under a fully global shuffle.
        """
        groups = {}
        for i, (label, shard_file, _row_idx) in enumerate(self.entries):
            groups.setdefault((label, shard_file), []).append(i)

        rng = random.Random(seed)
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        my_group_keys = group_keys[worker_id::num_workers] if num_workers > 1 else group_keys

        ordered = []
        for key in my_group_keys:
            idxs = groups[key]
            rng.shuffle(idxs)
            ordered.extend(idxs)
        return ordered

    def decode(self, image_bytes, label):
        """Turn raw jpeg bytes + original label id into a (tensor, idx) pair,
        using the same transform/class remap regardless of whether the bytes
        came from disk or were just streamed live."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image)
        return tensor, self.class_to_idx[label]

    def add(self, label, image_bytes):
        """Queue a freshly streamed (label, jpeg_bytes) sample to be buffered
        in RAM and eventually flushed to the on-disk cache in bulk. Skips
        images whose hash is already known to be cached. Non-blocking;
        best-effort (drops the sample on backpressure rather than blocking
        the training loop)."""
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        known = self._known_hashes.setdefault(label, set())
        if image_hash in known:
            return
        known.add(image_hash)
        try:
            self._write_queue.put_nowait((label, image_hash, image_bytes))
        except queue.Full:
            known.discard(image_hash)  # allow a future add() to retry

    def flush(self):
        """Wait for all queued adds to be buffered, then flush every
        partially-filled per-label buffer to disk as a final (possibly
        smaller than `flush_every`) shard.

        Call this once no more add() calls will be made (e.g. at the end of
        an epoch) - otherwise images still sitting in RAM buffers are lost
        when the process/worker exits.
        """
        self._write_queue.join()
        for label in list(self._buffers.keys()):
            self._flush_label(label)

    def _writer_loop(self):
        while True:
            label, image_hash, image_bytes = self._write_queue.get()
            try:
                self._buffers.setdefault(label, []).append(image_bytes)
                self._pending_hashes.setdefault(label, []).append(image_hash)
                if len(self._buffers[label]) >= self.flush_every:
                    self._flush_label(label)
            finally:
                self._write_queue.task_done()

    def _flush_label(self, label):
        buffer = self._buffers.get(label)
        pending_hashes = self._pending_hashes.get(label)
        if not buffer:
            return
        try:
            label_dir = os.path.join(self.cache_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)

            # Random shard filename: occasional duplicate images across
            # restarts/concurrent workers are acceptable (hash dedup is
            # best-effort, not cross-process-synchronized).
            shard_file = f"shard_{uuid.uuid4().hex}.arrow"
            shard_path = os.path.join(label_dir, shard_file)
            feather.write_feather(pa.table({"image": buffer}), shard_path)

            new_rows = pa.table({
                "hash": pending_hashes,
                "shard_file": [shard_file] * len(pending_hashes),
                "row_idx": list(range(len(pending_hashes))),
            })
            hashes_path = os.path.join(label_dir, HASHES_FILENAME)
            if os.path.exists(hashes_path):
                try:
                    new_rows = pa.concat_tables([feather.read_table(hashes_path), new_rows])
                except (pa.ArrowInvalid, OSError):
                    pass
            feather.write_feather(new_rows, hashes_path)
        except OSError:
            pass
        finally:
            self._buffers[label] = []
            self._pending_hashes[label] = []

