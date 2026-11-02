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

    # Optimisation (SGD + StepLR, as in Krizhevsky et al. 2012)
    "max_epochs": 100,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_step_size": 30,
    "lr_gamma": 0.1,

    # Throughput on a "bigger GPU" PC. 96 fits comfortably in 10-12 GB at 227px with
    # bf16; raise to 128/192 on a 16-24 GB card and keep lr proportional.
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

# CPU / quick-verification variant: same recipe, drastically less work. Nothing here
# may touch the optimisation hyperparameters.
SMOKE_OVERRIDES = {
    "train_samples": 1024,
    "val_samples": 256,
    "num_classes": 10,
    "max_epochs": 2,
    "batch_size": 16,
    "eval_batch_size": 16,
    "precision": "32",
    "accelerator": "cpu",
    "num_workers": 2,
    "shuffle_buffer": 512,
    "version_name": "alexnet_smoke",
}
# ---------------------------------------------------------------------------


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


def main():
    # --smoke is intentionally the only accepted argument.
    smoke = "--smoke" in sys.argv[1:]
    cfg = dict(CONFIG)
    if smoke:
        cfg.update(SMOKE_OVERRIDES)
        print("=== SMOKE TEST MODE: reduced sizes, identical recipe ===")
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
    print(f"{'checkpoints':>12} = {cfg['checkpoint_path']}")
    print(f"{'tensorboard':>12} = {cfg['log_path']}")
    n = cfg["num_classes"]
    print(f"Chance-level accuracy for {n} classes is {100.0 / n:.2f}%; "
          "val_acc must clearly exceed it.")
    print("================================================")
    return trainer


if __name__ == "__main__":
    main()
