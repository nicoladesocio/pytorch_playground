import argparse
import os

import model
from utils.model_utils import get_latest_checkpoint
from custum_datasets.streaming_imagenet_dataset import StreamingImageNetSubsetDataset, pick_target_classes
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


if __name__ == "__main__":
    argparse_argument_parser = argparse.ArgumentParser(description="Train GoogLeNet model")
    argparse_argument_parser.add_argument("--version_name", type=str, default="googlenet_v1", help="Experiment/version name")
    argparse_argument_parser.add_argument("--num_classes", type=int, default=64, help="Number of classes for GoogLeNet")
    argparse_argument_parser.add_argument("--checkpoint_path", type=str, default="/workspace/GoogLeNet/checkpoints", help="Path to model checkpoint")
    argparse_argument_parser.add_argument("--train_samples", type=int, default=51200, help="Number of training samples (~0.5% of ImageNet)")
    argparse_argument_parser.add_argument("--val_samples", type=int, default=2048, help="Number of validation samples (~0.5% of ImageNet)")
    argparse_argument_parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    argparse_argument_parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint_path")
    args = argparse_argument_parser.parse_args()

    # Build model. If --resume is set, full trainer/optimizer/scheduler state is restored
    # by passing ckpt_path to trainer.fit(...) below.
    checkpoint_path = None
    target_classes = None
    lightning_model = None
    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_path, model_prefix="googlenet_epoch")
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            # Load model to restore target_classes from hparams
            lightning_model = model.GoogLeNetLightning.load_from_checkpoint(checkpoint_path)
            target_classes = lightning_model.hparams.get('target_classes', None)
            print(f"Restored target_classes from checkpoint: {target_classes}")
        else:
            print("--resume was set, but no checkpoint was found. Starting from scratch.")
            checkpoint_path = None

    # Streaming datasets: no blocking pre-download. Cached images (from a
    # previous run) are read straight off disk; anything else is pulled from
    # the live HF stream and cached in the background as training proceeds.
    if target_classes is None:
        target_classes = pick_target_classes("imagenet-1k", args.num_classes)
    target_classes = sorted(target_classes)

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

    # Fresh run: instantiate after target_classes are finalized so they are checkpointed.
    if lightning_model is None:
        lightning_model = model.GoogLeNetLightning(num_classes=args.num_classes, target_classes=target_classes)

    # IterableDataset: no shuffle=/sampler= (dataset already shuffles internally)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True)
    
    print("\nModel and dataset ready for training!")

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="auto",
        devices="auto",
        precision="32",
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        logger=TensorBoardLogger(save_dir="/workspace/GoogLeNet/logs", name=args.version_name),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=-1,
                every_n_epochs=1,
                dirpath=args.checkpoint_path,
                filename="googlenet_epoch_{epoch:02d}"
            ),
            StochasticWeightAveraging(
                swa_lrs=1e-4,              # Usually matches your starting LR
                swa_epoch_start=0.8,       # Start averaging in the last 20% of training
                annealing_epochs=2
            )
        ]
    )
    
    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=checkpoint_path)



    


