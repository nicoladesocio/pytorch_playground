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
- `__getitem__` looks up which shard a row lives in and reads it via
  memory-mapped Arrow (`feather.read_table(..., memory_map=True)`): the OS
  lazily pages in only the bytes actually touched, instead of us eagerly
  loading whole shards into RAM. This also means the underlying pages are
  shared (via the OS page cache) across DataLoader worker *processes* that
  happen to mmap the same shard file - no custom cross-process cache needed.
- A background thread (`request_prefetch`/`iter_cached`) warms upcoming
  shards' mmap pages a couple of shards ahead of consumption, hiding the
  cold-first-touch page-fault latency behind whatever the training loop is
  doing with the current shard.

Important: because random access still means one page fault per touched
page, callers should iterate in *shard-grouped* order (see `iter_cached()`)
rather than fully-randomizing the global index space - interleaving reads
across many shards would defeat prefetching and cause more scattered,
smaller reads than necessary.

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
    `cache_dir/<label>/*.arrow`, with memory-mapped, prefetched shard
    loading."""

    def __init__(self, cache_dir, target_classes, transform, flush_every=256, prefetch_lookahead=2):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.target_classes = list(target_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.target_classes)}
        self.transform = transform
        self.flush_every = flush_every
        self.prefetch_lookahead = prefetch_lookahead

        # Memory-mapped shard tables, keyed by (label, shard_file). No
        # eviction needed: mmap only reserves address space up front, actual
        # physical pages are faulted in (and can be reclaimed by the OS
        # under memory pressure) lazily, per access.
        self._shard_tables = {}

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

        # Background thread that warms (memory-maps + touches) shards ahead
        # of when iter_cached() will actually need them.
        self._prefetch_queue = queue.Queue(maxsize=prefetch_lookahead + 1)
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

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

    def _load_shard(self, key):
        """Return the memory-mapped Arrow table for shard `key` (label,
        shard_file), mapping it (cheap: no data copy) on first access."""
        table = self._shard_tables.get(key)
        if table is None:
            label, shard_file = key
            shard_path = os.path.join(self.cache_dir, str(label), shard_file)
            table = feather.read_table(shard_path, memory_map=True)
            self._shard_tables[key] = table
        return table

    def _read_from_shard(self, label, shard_file, row_idx):
        table = self._load_shard((label, shard_file))
        return table.column("image")[row_idx].as_py()

    def request_prefetch(self, key):
        """Hint that shard `key` (label, shard_file) will be needed soon; a
        background thread will memory-map it (and touch a row to fault in
        its pages) ahead of time. Non-blocking/best-effort."""
        try:
            self._prefetch_queue.put_nowait(key)
        except queue.Full:
            pass

    def _prefetch_loop(self):
        while True:
            key = self._prefetch_queue.get()
            try:
                table = self._load_shard(key)
                if table.num_rows:
                    # Touch a row so the kernel actually faults in the pages
                    # now, rather than only reserving the mapping.
                    table.column("image")[0].as_py()
            except OSError:
                pass
            finally:
                self._prefetch_queue.task_done()

    def iter_cached(self, seed, worker_id=0, num_workers=1):
        """Yield (tensor, label_idx) pairs from the cache in shard-grouped
        shuffled order: whole shards are assigned to this worker (never
        split across workers), shard visit order and within-shard order are
        shuffled, but two different shards' rows are never interleaved.
        While consuming shard i, requests background prefetching of shard
        i + prefetch_lookahead so its mmap pages are already warm by the
        time we reach it.
        """
        groups = {}
        for label, shard_file, row_idx in self.entries:
            groups.setdefault((label, shard_file), []).append(row_idx)

        rng = random.Random(seed)
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        my_group_keys = group_keys[worker_id::num_workers] if num_workers > 1 else group_keys

        for i in range(min(self.prefetch_lookahead, len(my_group_keys))):
            self.request_prefetch(my_group_keys[i])

        for i, key in enumerate(my_group_keys):
            ahead_idx = i + self.prefetch_lookahead
            if ahead_idx < len(my_group_keys):
                self.request_prefetch(my_group_keys[ahead_idx])

            label, shard_file = key
            row_idxs = list(groups[key])
            rng.shuffle(row_idxs)
            for row_idx in row_idxs:
                image_bytes = self._read_from_shard(label, shard_file, row_idx)
                yield self.decode(image_bytes, label)

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

