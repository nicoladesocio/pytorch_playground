"""Shared runtime setup: deterministic seeding and portable default paths.

Both training scripts import this so that "same command twice => same numbers" is a
property of the code, not of luck. Without seeding, ``random.sample`` in the class
selection, DataLoader worker shuffling and weight init all differ run to run, which
makes every convergence claim in ``repos_log/`` unverifiable.
"""

import os
import random

import numpy as np
import torch

# Repo root = .../pytorch_playground (this file lives in <root>/utils/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root():
    return _REPO_ROOT


def default_path(*parts):
    """Repo-relative default path, overridable per-run via the CLI flags.

    Derived from the repo root rather than a hardcoded ``/workspace/...`` so the
    scripts run on a plain PC and inside the dev container alike.
    """
    return os.path.join(_REPO_ROOT, *parts)


def seed_everything(seed=42, deterministic=False):
    """Seed python/numpy/torch and configure DataLoader worker behaviour.

    ``deterministic=True`` additionally enables ``torch.use_deterministic_algorithms``.
    It is off by default because some conv/LRN backward kernels then raise on CUDA
    unless ``CUBLAS_WORKSPACE_CONFIG`` is set, which would make the "just run it"
    experience fragile.
    """
    import lightning as pl

    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            torch.use_deterministic_algorithms(True)
    return seed


def available_shm_bytes():
    """Size of ``/dev/shm`` in bytes, or ``None`` where the concept does not apply.

    DataLoader workers hand tensors to the parent process through POSIX shared
    memory. Docker's default ``--shm-size`` is 64 MB, which a single worker carrying
    a batch of 224px float32 images (~20 MB per tensor, several in flight) can
    exhaust, producing the infamous
    ``unable to allocate shared memory ... No space left on device``.
    """
    if os.name != "posix":
        return None
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return None
    return st.f_bsize * st.f_blocks


def estimate_worker_shm_bytes(batch_size, image_size, channels=3, in_flight=4):
    """Rough shared-memory footprint of one DataLoader worker.

    Deliberately pessimistic: a float32 image batch plus a few buffers in flight.
    Used only to decide whether to warn/clamp ``num_workers``.
    """
    per_batch = int(batch_size) * channels * int(image_size) * int(image_size) * 4
    return max(1, per_batch * in_flight)


def count_available_workers(requested, bytes_per_worker=0):
    """Clamp ``num_workers`` to something the machine can actually provide.

    The hardcoded values (8 train / 2 val) oversubscribe small containers, which
    shows up as worker restarts and inflated epoch times rather than an error.

    When ``bytes_per_worker`` is given, the limit is also driven by ``/dev/shm``:
    a container started with the default 64 MB shm cannot support multi-worker
    loading of image batches at all, and failing fast with this message beats a
    cryptic ``No space left on device`` from deep inside the DataLoader.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        cpus = os.cpu_count() or 1
    limit = max(1, cpus - 1)

    shm = available_shm_bytes()
    if shm is not None and bytes_per_worker > 0:
        shm_workers = int(shm // bytes_per_worker)
        if shm_workers < limit:
            print(f"[INFO] /dev/shm is {shm / 1e6:.0f} MB; a batch of this size needs "
                  f"~{bytes_per_worker / 1e6:.0f} MB per DataLoader worker, so "
                  f"num_workers is capped at {shm_workers}. Start the container with "
                  "`--shm-size=4g` (or run natively) to use more workers.")
            limit = shm_workers

    return max(0, min(int(requested), limit))


def resolve_accelerator(requested="auto"):
    """Return ``(accelerator, devices)`` for Lightning, tolerating a missing GPU.

    ``accelerator="auto"`` already does this, but an explicit request for "gpu" on a
    CPU-only box must fail with an explanation instead of Lightning's stack trace.
    """
    if requested == "cpu":
        return "cpu", 1
    if requested == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--accelerator gpu was requested but torch.cuda.is_available() is False. "
                "Check the driver (`nvidia-smi`) and that the installed torch wheel matches "
                "your GPU architecture; use --accelerator auto to fall back to CPU."
            )
        return "gpu", torch.cuda.device_count()
    return "auto", "auto"
