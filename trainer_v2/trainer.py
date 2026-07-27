import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from trainer_v2.load_module import load_module
import os

DATASET_CACHE_ROOT = "/workspace/Data/checkpoints/"

TRAINER_REGISTRY = {
    "default": {
        "description": "Default trainer configuration with 10 epochs and TensorBoard logging.",
        "module": "trainer_v2.trainer"
    }
}

def trainer(trainer_name, model_name, run_name):
    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{trainer_name}' not found in registry.")
    
    module_path = TRAINER_REGISTRY[trainer_name]["module"]
    class_name = "default"
    
    try:
        trainer_instance = load_module(module_path, class_name, model_name=model_name, run_name=run_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not load trainer class '{class_name}' from module '{module_path}'."
        ) from exc 
    
    return trainer_instance

def default(model_name, run_name):
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(f"tb_logs/{model_name}", name=f"{run_name}"),
        callbacks=[
            ModelCheckpoint(monitor="val_loss",
                            dirpath=os.path.join(DATASET_CACHE_ROOT, model_name),
                            filename=f"{run_name}-{{epoch:03d}}-",
                            save_top_k=1, mode="min"),
        ]
    )

    return trainer