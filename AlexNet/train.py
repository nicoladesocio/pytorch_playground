"""AlexNet training entry point.

Everything happens inside ``main()`` under an ``if __name__ == "__main__"`` guard.
The previous version built the ``pl.Trainer`` and called ``trainer.fit(...)`` at
module scope, so importing this file (which ``trainer/trainer.py`` does when it
inspects a model directory) started a full training run as a side effect.

Run it directly -- no arguments are required for a sensible default run:

    python AlexNet/train.py

For the frozen, "this is the real experiment" configuration use the parameterless
``AlexNet/train_final.py`` instead.
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import model
from utils.datasets_utils import (
    DEFAULT_DATASET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_SHUFFLE_BUFFER,
    load_dataset_subset,
    create_subset_sampler,
    create_eval_sampler,
)
from utils.model_utils import get_latest_checkpoint, save_model_checkpoint
from utils.runtime import (
    default_path,
    seed_everything,
    count_available_workers,
    estimate_worker_shm_bytes,
    resolve_accelerator,
)
from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train AlexNet on an ImageNet-1k subset")
    parser.add_argument("--version_name", type=str, default="alexnet_v1",
                        help="Experiment/version name (tensorboard run name)")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET,
                        help=f"HuggingFace dataset id (default: {DEFAULT_DATASET}; gated, needs HF_TOKEN)")
    parser.add_argument("--cache_root", type=str, default=DEFAULT_CACHE_ROOT,
                        help="Where downloaded dataset subsets are cached")
    parser.add_argument("--num_classes", type=int, default=64,
                        help="Number of classes to train on (subset of the dataset's classes)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"],
                        help="Optimizer. AlexNet's paper value is SGD+momentum; Adam needs "
                             "a much smaller --lr (~1e-4) than SGD (0.01).")
    parser.add_argument("--scheduler", type=str, default="step",
                        choices=["step", "cosine", "none"],
                        help="LR schedule: step (paper), cosine anneal, or none.")
    parser.add_argument("--image_size", type=int, default=227,
                        help="Input resolution; the classifier's flatten size is derived from it")
    parser.add_argument("--checkpoint_path", type=str, default=default_path("AlexNet", "checkpoints"),
                        help="Path to model checkpoints")
    parser.add_argument("--log_path", type=str, default=default_path("Data", "logs"),
                        help="TensorBoard log root")
    parser.add_argument("--train_samples", type=int, default=51200,
                        help="Number of training samples cached from the dataset (~4% of ImageNet)")
    parser.add_argument("--train_samples_per_epoch", type=int, default=None,
                        help="Optimizer samples drawn per epoch. Defaults to one pass over the "
                             "in-class cached images. Set it higher to lengthen epochs when the "
                             "cached subset is small (a small subset spread over 1000 classes "
                             "leaves only a handful of images per class, which otherwise yields "
                             "1-2 batches per epoch and looks like a flat loss). Draws with "
                             "replacement above the in-class count.")
    parser.add_argument("--val_samples", type=int, default=2048,
                        help="Number of validation samples cached from the dataset")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for validation (defaults to --batch_size)")
    parser.add_argument("--max_epochs", type=int, default=100, help="Training length in epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="SGD weight decay")
    parser.add_argument("--lr_step_size", type=int, default=30, help="StepLR epoch interval")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="StepLR decay factor")
    parser.add_argument("--precision", type=str, default="32",
                        help="Lightning precision, e.g. 32, 16-mixed, bf16-mixed (use 16-mixed on a real GPU)")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Desired DataLoader workers; clamped to the CPUs actually available")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    parser.add_argument("--shuffle_buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER,
                        help="Streaming shuffle buffer; larger = better mixing but more RAM "
                             "and a longer stall before the first sample")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic algorithms (slower, may warn on some CUDA kernels)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in --checkpoint_path")
    parser.add_argument("--limit_train_batches", type=float, default=None,
                        help="Fraction (0..1] or count of train batches per epoch -- for smoke tests")
    parser.add_argument("--limit_val_batches", type=float, default=None,
                        help="Fraction (0..1] or count of val batches per epoch -- for smoke tests")
    parser.add_argument("--log_grad_norms", action="store_true", default=True,
                        help="Log aggregate gradient norm to TensorBoard every 50 steps")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    seed_everything(args.seed, deterministic=args.deterministic)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    print("\nLoading ImageNet subset...")
    train_dataset, validation_dataset, _cached_classes = load_dataset_subset(
        dataset_name=args.dataset_name,
        cache_root=args.cache_root,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer,
    )

    # If a checkpoint exists, restore target_classes from it so the sampler uses
    # exactly the same class subset the model was trained on.
    checkpoint_path = None
    target_classes = None
    lightning_model = None

    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_path)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            lightning_model = model.AlexNetLightning.load_from_checkpoint(checkpoint_path)
            target_classes = lightning_model.hparams.target_classes
            # Re-checkpointing must not silently change the experiment.
            for key in ("num_classes", "image_size"):
                saved = lightning_model.hparams.get(key)
                if saved is not None and saved != getattr(args, key):
                    raise ValueError(
                        f"--{key}={getattr(args, key)} conflicts with the checkpoint's "
                        f"{key}={saved}. Use a fresh --checkpoint_path for a new experiment."
                    )
        else:
            print("--resume was set, but no checkpoint was found. Starting from scratch.")
            checkpoint_path = None
    else:
        print("No checkpoint requested, starting fresh training run")

    # Single source of truth for the class count: the sampler, the dataset remapping
    # and the model head all consume the same target_classes list. The previous code
    # asked the sampler for max_num_classes=64 while building a model with
    # args.num_classes, so --num_classes != 64 produced a head whose outputs could
    # never match the remapped labels.
    train_sampler, target_classes = create_subset_sampler(
        train_dataset, max_num_classes=args.num_classes, target_classes=target_classes,
        seed=args.seed, val_dataset=validation_dataset,
        num_samples=args.train_samples_per_epoch,
    )
    target_classes = sorted(target_classes)
    num_classes = len(target_classes)
    if num_classes != args.num_classes:
        print(f"[WARN] Training on {num_classes} classes instead of the requested "
              f"{args.num_classes} (limited by cached data).")

    val_sampler = create_eval_sampler(validation_dataset, target_classes)

    train_dataset = ImageNetSubsetDataset(
        train_dataset, image_size=args.image_size, is_training=True,
        target_classes=target_classes)
    validation_dataset = ImageNetSubsetDataset(
        validation_dataset, image_size=args.image_size, is_training=False,
        target_classes=target_classes)

    if lightning_model is None:
        print(f"Initializing model with {num_classes} classes at {args.image_size}px...")
        lightning_model = model.AlexNetLightning(
            num_classes=num_classes,
            target_classes=target_classes,
            image_size=args.image_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            log_grad_norms=args.log_grad_norms,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
        )
    print(args)

    train_workers = count_available_workers(
        args.num_workers,
        bytes_per_worker=estimate_worker_shm_bytes(args.batch_size, args.image_size))
    eval_batch_size = args.eval_batch_size or args.batch_size
    eval_workers = count_available_workers(
        min(2, args.num_workers),
        bytes_per_worker=estimate_worker_shm_bytes(eval_batch_size, args.image_size))
    # drop_last must never be able to empty an epoch: with the old
    # drop_last=True on validation, a val set smaller than one batch produced zero
    # val batches and therefore no val_loss to monitor -- ModelCheckpoint then
    # silently saved nothing.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=train_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=train_workers > 0)
    val_loader = DataLoader(
        validation_dataset, batch_size=eval_batch_size, sampler=val_sampler,
        num_workers=eval_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=eval_workers > 0)

    print("\nModel and dataset ready for training!")

    accelerator, devices = resolve_accelerator(args.accelerator)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=2,
        logger=TensorBoardLogger(save_dir=args.log_path, name=args.version_name),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=-1,
                every_n_epochs=1,
                dirpath=args.checkpoint_path,
                filename="alexnet_epoch_{epoch:02d}",
            ),
            # Convergence guard: stop early if validation stops improving, so a
            # mis-specified run cannot silently burn hours on a bigger GPU.
            EarlyStopping(
                monitor="val_loss",
                patience=max(10, args.max_epochs // 5),
                mode="min",
                divergence_threshold=50.0,
            ),
        ],
    )
    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=checkpoint_path)

    save_model_checkpoint(lightning_model, args.checkpoint_path, filename="alexnet_final.ckpt")
    return trainer


if __name__ == "__main__":
    main()
