"""AlexNet -- the real training run. NO ARGUMENTS.

    python AlexNet/train_final.py

This file is the single source of truth for the AlexNet experiment. Every
hyperparameter is baked in below; there is no argparse, no environment sniffing and
no "it worked on my machine" drift. If you want a different configuration, copy this
file and change the constants -- do not add flags, because the whole point is that
the run the human launches is exactly the run that was validated.

Sizing intent (see repos_log/goal.md): the run must stay *small* but be sized for a
PC with a proper GPU. The defaults below are the "real" experiment for a >=10 GB
CUDA card. On a CPU-only box, use the smoke-test path instead:

    python AlexNet/train_final.py --smoke

(--smoke is the one concession to argparse; it only shrinks the run, it can never
change the recipe, so the baked-in configuration stays authoritative.)
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

# ---------------------------------------------------------------------------
# BAKED-IN CONFIGURATION -- edit here, never on the command line.
# ---------------------------------------------------------------------------
CONFIG = {
    # Data
    "dataset_name": "ILSVRC/imagenet-1k",   # gated: requires HF_TOKEN in the environment
    "cache_root": None,                     # None -> repo-relative Data/datasets
    "train_samples": 51200,                 # ~4% of the 1.28M train images
    "val_samples": 2048,
    "num_classes": 64,                      # chance level = 1/64 = 1.56%
    "image_size": 227,                      # AlexNet paper; flatten size is derived

    # Epoch length. NOT the same knob as train_samples: 51200 is what was *downloaded*,
    # and choose_target_classes() only keeps the 64 best-represented of the 1000 classes
    # -- 4153 rows on the canonical cache, i.e. a default epoch is 65 steps at batch 64.
    # HANDOFF_003's "plateaued after 18 epochs" was 18 x 65 = ~1170 optimiser steps, far
    # too few to move the weights. Drawing 16384 per epoch (with replacement above the
    # in-class count) is the value both converged runs used:
    # repos_log/evidences/E030-googlenet-converged-run.md and
    # repos_log/evidences/E031-alexnet-adam-converges-sgd-pinned-at-chance.md.
    "train_samples_per_epoch": 16384,

    # Optimisation. Adam is the recipe that was *observed* to converge (val_acc 24.8%
    # against a 1.56% chance level, E031). The paper's SGD + StepLR is NOT baked in:
    # train.py passes gradient_clip_val=1.0 and AlexNet's total gradient norm at
    # initialisation is 26.9, so every SGD step is scaled by 0.037 -- 21 epochs of SGD
    # lr=0.1 left train_acc at chance (4.16 +/- 0.01 val_loss, i.e. exactly ln(64)).
    # That is a measurement about the clip, not a verdict on SGD. A fair SGD run needs
    # the clip raised above the init norm first; see card P2-25 and AlexNet/train.py's
    # --gradient_clip_val.
    "max_epochs": 100,
    "optimizer": "adam",
    "lr": 3e-4,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_step_size": 30,
    "lr_gamma": 0.1,
    "scheduler": "cosine",

    # Throughput on a "bigger GPU" PC. 96 fits comfortably in 10-12 GB at 227px with
    # bf16; raise to 128/192 on a 16-24 GB card and keep lr proportional.
    # CAVEAT: E031 measured lr=3e-4 at batch_size=32 on a 6 GB card. 96 here is an
    # unmeasured extrapolation -- lr was never swept at this batch size (P2-26).
    "batch_size": 96,
    "eval_batch_size": 128,
    "precision": "16-mixed",                # bf16/fp16 autocast -- the single biggest
                                            # speed/VRAM win on a modern GPU
    "accelerator": "auto",
    "num_workers": 8,

    # Reproducibility
    "seed": 42,
    "deterministic": False,
    "shuffle_buffer": 10_000,               # 512 GB-RAM-class box: mixing quality wins

    # Bookkeeping
    "version_name": "alexnet_final",
    "checkpoint_path": None,                # None -> repo-relative AlexNet/checkpoints
    "log_path": None,                       # None -> repo-relative Data/logs
    "log_grad_norms": True,
}

# Quick-verification variant: same recipe, drastically less work. Nothing here may
# touch the optimisation hyperparameters.
#
# `accelerator`/`precision` are deliberately NOT pinned to cpu/32 here -- see
# _smoke_overrides(). Pinning them meant a GPU box silently ran its smoke test on CPU,
# which is how "the GPU is unusable" survived three handoffs unchallenged.
SMOKE_OVERRIDES = {
    "train_samples": 1024,
    "val_samples": 256,
    "num_classes": 10,
    "max_epochs": 2,
    "batch_size": 16,
    "eval_batch_size": 16,
    "num_workers": 2,
    "shuffle_buffer": 512,
    # Back to one pass over the in-class samples. The point of the long epoch in CONFIG
    # is to give a real run enough optimiser steps; a smoke test only has to execute,
    # and 16384 samples at batch 16 would make it 1024 steps per epoch.
    "train_samples_per_epoch": None,
    "version_name": "alexnet_smoke",
}
# ---------------------------------------------------------------------------


def _smoke_overrides(cfg):
    """Resolve the smoke variant against the hardware actually present.

    The point of --smoke is "prove the pipeline works, as fast as possible, without
    changing the recipe". Running it on the fastest device available is therefore the
    faithful reading, and it is what makes a GPU smoke run and a CPU smoke run the same
    experiment rather than two different ones.

    On a GPU we keep CONFIG's precision (so the mixed-precision path the human will
    actually run is the path being tested); on CPU we must fall back to 32, which
    Lightning requires. Batch size stays small either way so the run fits a 6 GB card.
    """
    import torch

    overrides = dict(SMOKE_OVERRIDES)
    if torch.cuda.is_available():
        overrides["accelerator"] = "gpu"
        overrides["precision"] = cfg["precision"]
    else:
        overrides["accelerator"] = "cpu"
        overrides["precision"] = "32"
    return overrides


def _apply_repo_defaults(cfg):
    from utils.runtime import default_path

    cfg = dict(cfg)
    if cfg["cache_root"] is None:
        cfg["cache_root"] = default_path("Data", "datasets")
    if cfg["checkpoint_path"] is None:
        cfg["checkpoint_path"] = default_path("AlexNet", "checkpoints")
    if cfg["log_path"] is None:
        cfg["log_path"] = default_path("Data", "logs")
    return cfg


def _describe_device(trainer):
    """Human-readable device string for the RESULT block.

    Without this, a handoff cannot tell a CPU result from a GPU result, which is
    precisely how a stale "CPU only" belief survived in AGENT_INSTRUCTIONS.md for
    three handoffs.
    """
    import torch

    # Lightning's Trainer has no `.device`; the placed device lives on the strategy.
    try:
        root = str(trainer.strategy.root_device)
    except Exception:
        root = "unknown"
    if root.startswith("cuda") and torch.cuda.is_available():
        return f"{root} ({torch.cuda.get_device_name(0)})"
    return root


def main():
    # --smoke is intentionally the only accepted argument; anything else is a typo and
    # must fail loudly rather than silently start a full run (see runtime.validate_final_args).
    from utils.runtime import validate_final_args

    smoke = validate_final_args(sys.argv[1:], "AlexNet/train_final.py")
    cfg = dict(CONFIG)
    if smoke:
        cfg.update(_smoke_overrides(cfg))
        print("=== SMOKE TEST MODE: reduced sizes, identical recipe ===")
        print(f"    device    = {cfg['accelerator']}  (chosen from torch.cuda.is_available())")
        print(f"    precision = {cfg['precision']}")
    cfg = _apply_repo_defaults(cfg)

    if cfg["dataset_name"] == "ILSVRC/imagenet-1k" and not (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print(
            "WARNING: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment.\n"
            "         ILSVRC/imagenet-1k is gated and the download will fail.\n"
            "         Accept the terms on https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
            "         then:  export HF_TOKEN=hf_...   (or  $env:HF_TOKEN='hf_...')\n"
            "         If a cached subset already exists under the cache root, this run\n"
            "         proceeds without network access.\n"
        )

    argv = []
    for key, value in cfg.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        argv.extend([f"--{key}", str(value)])

    import train
    trainer = train.main(argv)

    # Print the numbers the definition-of-done is judged on, so a human can eyeball
    # success without opening TensorBoard.
    metrics = trainer.callback_metrics
    print("\n==================== RESULT ====================")
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        if key in metrics:
            print(f"{key:>12} = {float(metrics[key]):.4f}")
    # Record what the run actually executed on. Without this, a handoff cannot tell a
    # CPU result from a GPU result, which is precisely how a stale "CPU only" belief
    # survived in AGENT_INSTRUCTIONS.md for three handoffs.
    print(f"{'device':>12} = {_describe_device(trainer)}")
    print(f"{'precision':>12} = {trainer.precision}")
    print(f"{'checkpoints':>12} = {cfg['checkpoint_path']}")
    print(f"{'tensorboard':>12} = {cfg['log_path']}")
    n = cfg["num_classes"]
    print(f"Chance-level accuracy for {n} classes is {100.0 / n:.2f}%; "
          "val_acc must clearly exceed it.")
    print("================================================")
    return trainer


if __name__ == "__main__":
    main()
