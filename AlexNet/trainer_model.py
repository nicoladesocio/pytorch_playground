try:
    from trainer.model_base import Model
except ModuleNotFoundError:
    from model_base import Model


class AlexNetModel(Model):
    name = "alexnet"
    train_script = "train.py"
    default_parameters = {
        "version_name": "alexnet_v1",
        "num_classes": 64,
        "checkpoint_path": "/workspace/AlexNet/checkpoints",
        "train_samples": 51200,
        "val_samples": 2048,
        "batch_size": 96,
    }
