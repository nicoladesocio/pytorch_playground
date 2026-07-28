import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import model
import argparse
from utils.model_utils import get_latest_checkpoint, save_model_checkpoint
from custum_datasets.streaming_imagenet_dataset import StreamingImageNetSubsetDataset, pick_target_classes
from torch.utils.data import DataLoader
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import gc
gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    argparse_argument_parser = argparse.ArgumentParser(description="Train AlexNet model")
    argparse_argument_parser.add_argument("--version_name", type=str, default="alexnet_v1", help="Experiment/version name")
    argparse_argument_parser.add_argument("--num_classes", type=int, default=64, help="Number of classes for AlexNet")
    argparse_argument_parser.add_argument("--checkpoint_path", type=str, default="/workspace/AlexNet/checkpoints", help="Path to model checkpoint")
    argparse_argument_parser.add_argument("--train_samples", type=int, default=51200, help="Number of training samples (~0.5% of ImageNet)")
    argparse_argument_parser.add_argument("--val_samples", type=int, default=2048, help="Number of validation samples (~0.5% of ImageNet)")
    argparse_argument_parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training")
    args = argparse_argument_parser.parse_args()

    # If a checkpoint exists, restore target_classes from it so training resumes
    # on the same class subset (exact images seen are allowed to differ).
    checkpoint_path = get_latest_checkpoint(args.checkpoint_path)
    print(f"Checkpoint path: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        lightning_model = model.AlexNetLightning.load_from_checkpoint(checkpoint_path)
        target_classes = lightning_model.hparams.target_classes
    else:
        print("No checkpoint found, starting fresh training run")
        target_classes = pick_target_classes("imagenet-1k", args.num_classes)

    # Streaming datasets: no blocking pre-download. Cached images (from a
    # previous run) are read straight off disk; anything else is pulled from
    # the live HF stream and cached in the background as training proceeds.
    train_dataset = StreamingImageNetSubsetDataset(
        dataset_name="imagenet-1k",
        target_classes=target_classes,
        num_samples=args.train_samples,
        cache_root="/workspace/Data/datasets/",
        split="train",
        image_size=227,
        is_training=True,
    )
    validation_dataset = StreamingImageNetSubsetDataset(
        dataset_name="imagenet-1k",
        target_classes=target_classes,
        num_samples=args.val_samples,
        cache_root="/workspace/Data/datasets/",
        split="validation",
        image_size=227,
        is_training=False,
    )

    if not (checkpoint_path and os.path.exists(checkpoint_path)):
        print("Initializing model...")
        # Pass target_classes so the mapping is saved in the checkpoint via save_hyperparameters()
        lightning_model = model.AlexNetLightning(num_classes=args.num_classes, target_classes=target_classes)
    print(args)

    # IterableDataset: no shuffle=/sampler= (dataset already shuffles internally)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True)
    
    print("\nModel and dataset ready for training!")

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    precision="32",
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    logger=TensorBoardLogger(save_dir="/workspace/Data/logs", name=args.version_name),
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=-1,
            every_n_epochs=1,
            dirpath=args.checkpoint_path,
            filename="alexnet_epoch_{epoch:02d}"
        )
    ]
)
trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=checkpoint_path)

save_model_checkpoint(lightning_model, args.checkpoint_path)
