try:
    from trainer.model_base import Model
except ModuleNotFoundError:
    from model_base import Model

try:
    from utils.runtime import default_path
except ModuleNotFoundError:  # loaded by importlib without the repo root on sys.path
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def default_path(*parts):
        return os.path.join(_ROOT, *parts)


class GoogLeNetModel(Model):
    name = "googlenet"
    train_script = "train.py"
    # Keys are forwarded verbatim as `--key value` by Model.build_train_command, so
    # every entry must be a real flag on GoogLeNet/train.py. Paths come from the repo
    # root instead of hardcoded /workspace/... so the registry also works outside the
    # dev container.
    default_parameters = {
        "version_name": "googlenet_v1",
        "dataset_name": "ILSVRC/imagenet-1k",
        "cache_root": default_path("Data", "datasets"),
        "num_classes": 64,
        "image_size": 227,
        "checkpoint_path": default_path("GoogLeNet", "checkpoints"),
        "log_path": default_path("GoogLeNet", "logs"),
        "train_samples": 51200,
        "val_samples": 2048,
        "batch_size": 64,
        "max_epochs": 60,
        "optimizer": "adam",
        "scheduler": "cosine",
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "eta_min": 0.000001,
        "aux_loss_weight": 0.3,
        "swa": True,
        "swa_lrs": 0.0001,
        "swa_epoch_start": 0.8,
        "annealing_epochs": 2,
        "precision": "32",
        "accelerator": "auto",
        "num_workers": 8,
        "seed": 42,
    }
