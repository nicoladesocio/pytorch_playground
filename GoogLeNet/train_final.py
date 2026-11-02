"""GoogLeNet -- the real training run. NO ARGUMENTS.

    python GoogLeNet/train_final.py

This file is the single source of truth for the GoogLeNet experiment. Every
hyperparameter is baked in below; there is no argparse and no environment sniffing,
so the run the human launches is exactly the run that was validated.

Sizing intent (see repos_log/goal.md): the run must stay *small* but be sized for a
PC with a proper GPU. The defaults below are the "real" experiment for a >=10 GB
CUDA card. On a CPU-only box, use the smoke-test path instead:

    python GoogLeNet/train_final.py --smoke

(--smoke is the one concession to argparse; it only shrinks the run, it can never
change the recipe.)
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
    "image_size": 227,

    # Optimisation. Adam was the configuration that actually trained in this repo;
    # the paper's SGD+StepLR is available via --optimizer sgd on train.py, but the
    # baked-in recipe is the one that was observed to converge.
    "max_epochs": 60,
    "optimizer": "adam",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "eta_min": 1e-6,
    "aux_loss_weight": 0.3,                 # Szegedy et al. 2015 use 0.3 per aux head

    # Stochastic weight averaging over the last 20% of training.
    "swa": True,
    "swa_lrs": 1e-4,
    "swa_epoch_start": 0.8,
    "annealing_epochs": 2,

    # Throughput on a "bigger GPU" PC. GoogLeNet is ~10.5M params, so 128 at 227px
    # with bf16 fits in ~10 GB; raise to 192/256 on a 16-24 GB card.
    "batch_size": 128,
    "eval_batch_size": 192,
    "precision": "16-mixed",
    "accelerator": "auto",
    "num_workers": 8,

    # Reproducibility
    "seed": 42,
    "deterministic": False,
    "shuffle_buffer": 10_000,

    # Bookkeeping
    "version_name": "googlenet_final",
    "checkpoint_path": None,                # None -> repo-relative GoogLeNet/checkpoints
    "log_path": None,                       # None -> repo-relative GoogLeNet/logs
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
    "swa": False,                           # SWA needs a long run to be meaningful
    "version_name": "googlenet_smoke",
}
# ---------------------------------------------------------------------------


def _apply_repo_defaults(cfg):
    from utils.runtime import default_path

    cfg = dict(cfg)
    if cfg["cache_root"] is None:
        cfg["cache_root"] = default_path("Data", "datasets")
    if cfg["checkpoint_path"] is None:
        cfg["checkpoint_path"] = default_path("GoogLeNet", "checkpoints")
    if cfg["log_path"] is None:
        cfg["log_path"] = default_path("GoogLeNet", "logs")
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
            elif key == "swa":
                argv.append("--no_swa")
            continue
        argv.extend([f"--{key}", str(value)])

    import train
    trainer = train.main(argv)

    metrics = trainer.callback_metrics
    print("\n==================== RESULT ====================")
    for key in ("train_loss", "train_loss_main", "train_acc", "val_loss", "val_acc"):
        if key in metrics:
            print(f"{key:>16} = {float(metrics[key]):.4f}")
    print(f"{'checkpoints':>16} = {cfg['checkpoint_path']}")
    print(f"{'tensorboard':>16} = {cfg['log_path']}")
    n = cfg["num_classes"]
    print(f"Chance-level accuracy for {n} classes is {100.0 / n:.2f}%; "
          "val_acc must clearly exceed it.")
    print("================================================")
    return trainer


if __name__ == "__main__":
    main()
