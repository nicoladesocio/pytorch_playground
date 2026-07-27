# PyTorch Playground

Experiments implementing classic CNN architectures from scratch (PyTorch + PyTorch
Lightning), plus a small CLI framework for registering and training them.

## Folder Guide

| Folder | Status | Purpose |
|---|---|---|
| [AlexNet/](AlexNet/) | WIP | AlexNet (2012) implementation. See [AlexNet/Status.md](AlexNet/Status.md) — training currently plateaus due to a vanishing-gradient issue that hasn't been resolved yet (Batch Norm, Adam, gradient clipping, Kaiming init all tried). |
| [GoogLeNet/](GoogLeNet/) | Active | GoogLeNet (2014) implementation with Inception modules + auxiliary classifiers. More stable than AlexNet; supports `--resume` from checkpoint. |
| [trainer/](trainer/) | Active | The current training CLI. Register a model folder (`trainer register`) then train it (`trainer train <Model>`) with layered parameter overrides. See [trainer/README.md](trainer/README.md). Each model folder plugs in via its own `trainer_model.py`. |
| [trainer_v2/](trainer_v2/) | **WIP / design draft** | Next-gen training orchestrator (datasets, model versions, pipelines, runs as first-class, reproducible entities). Registries (`model.py`, `dataset.py`, `trainer.py`) are currently hardcoded stubs, not a finished implementation. See [trainer_v2/README.md](trainer_v2/README.md) for the design. Not yet a replacement for `trainer/`. |
| [utils/](utils/) | Active | Shared helpers used across model folders: dataset subset loading/caching (`datasets_utils.py`), checkpoint lookup (`model_utils.py`), misc helpers (`helpers.py`). |
| [custum_datasets/](custum_datasets/) | Active (minor) | `ImagesFolderDataset` — loads a plain folder of images for ad-hoc inference (used by `AlexNet/run.py`). |
| [Data/](Data/) | Storage | Local data/output mount. `Data/datasets/` caches downloaded dataset subsets; `Data/images/` is a scratch space for debug images. Both are gitignored (empty placeholders only). |

## Root Files

- `pyproject.toml` — project metadata + dependencies (torch, lightning, datasets, etc.). Packages built: `AlexNet`, `GoogLeNet`, `utils`, `custum_datasets`, `trainer_v2`.
- `Dockerfile` / `.dockerignore` / `.devcontainer/` — containerized dev environment (CUDA 12.8, PyTorch 2.10).

## Typical Workflow

```bash
# Register a model (once, or after changing its trainer_model.py)
python3 trainer register --model-name GoogLeNet --model-dir GoogLeNet --force

# Train with defaults, or override params
python3 trainer train GoogLeNet
python3 trainer train GoogLeNet --version-name googlenet_exp_01 --batch-size 128

# Dry run to see the resolved command without training
python3 trainer train GoogLeNet --dry-run
```

## Known Issues / In-Progress Work

- **AlexNet** training is not converging (see [AlexNet/Status.md](AlexNet/Status.md)) — treat as an open experiment, not a working baseline.
- **trainer_v2** is a design draft with hardcoded registries; use **trainer/** for actual training runs until it's completed.
