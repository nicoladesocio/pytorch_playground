"""GoogLeNet training entry point.

Everything happens inside ``main()`` under an ``if __name__ == "__main__"`` guard so
that importing this module (``trainer/trainer.py`` does) never starts a training run.

Run it directly -- no arguments are required for a sensible default run:

    python GoogLeNet/train.py

For the frozen, "this is the real experiment" configuration use the parameterless
``GoogLeNet/train_final.py`` instead.
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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
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
    parser = argparse.ArgumentParser(description="Train GoogLeNet on an ImageNet-1k subset")
    parser.add_argument("--version_name", type=str, default="googlenet_v1",
                        help="Experiment/version name (tensorboard run name)")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET,
                        help=f"HuggingFace dataset id (default: {DEFAULT_DATASET}; gated, needs HF_TOKEN)")
    parser.add_argument("--cache_root", type=str, default=DEFAULT_CACHE_ROOT,
                        help="Where downloaded dataset subsets are cached")
    parser.add_argument("--num_classes", type=int, default=64,
                        help="Number of classes to train on (subset of the dataset's classes)")
    parser.add_argument("--image_size", type=int, default=227,
                        help="Input resolution; aux-head flatten sizes are derived from it")
    parser.add_argument("--checkpoint_path", type=str, default=default_path("GoogLeNet", "checkpoints"),
                        help="Path to model checkpoints")
    parser.add_argument("--log_path", type=str, default=default_path("GoogLeNet", "logs"),
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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for validation (defaults to --batch_size)")
    parser.add_argument("--max_epochs", type=int, default=60, help="Training length in epochs")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Adam was the working default; SGD matches the 2015 paper")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (sgd only)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Cosine schedule floor")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"],
                        help="LR schedule. Cosine anneals to --eta_min over --max_epochs.")
    parser.add_argument("--aux_loss_weight", type=float, default=0.3,
                        help="Weight of each auxiliary classifier loss (0.3 per the paper)")
    parser.add_argument("--swa", action="store_true", default=True,
                        help="Enable StochasticWeightAveraging (default on, as before)")
    parser.add_argument("--no_swa", dest="swa", action="store_false", help="Disable SWA")
    parser.add_argument("--swa_lrs", type=float, default=1e-4, help="SWA learning rate")
    parser.add_argument("--swa_epoch_start", type=float, default=0.8,
                        help="Fraction of training after which SWA starts averaging")
    parser.add_argument("--annealing_epochs", type=int, default=2, help="SWA annealing length")
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
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    seed_everything(args.seed, deterministic=args.deterministic)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Build model. If --resume is set, full trainer/optimizer/scheduler state is
    # restored by passing ckpt_path to trainer.fit(...) below.
    checkpoint_path = None
    target_classes = None
    lightning_model = None
    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_path, model_prefix="googlenet_epoch")
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            lightning_model = model.GoogLeNetLightning.load_from_checkpoint(checkpoint_path)
            target_classes = lightning_model.hparams.get('target_classes', None)
            print(f"Restored target_classes from checkpoint: {target_classes}")
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

    print("\nLoading ImageNet subset...")
    train_dataset, validation_dataset, _cached_classes = load_dataset_subset(
        dataset_name=args.dataset_name,
        cache_root=args.cache_root,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )

    # Single source of truth for the class count: sampler, label remapping and model
    # heads all consume the same target_classes list.
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

    # Fresh run: instantiate after target_classes are finalized so they are checkpointed.
    if lightning_model is None:
        print(f"Initializing model with {num_classes} classes at {args.image_size}px...")
        lightning_model = model.GoogLeNetLightning(
            num_classes=num_classes,
            target_classes=target_classes,
            image_size=args.image_size,
            aux_loss_weight=args.aux_loss_weight,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eta_min=args.eta_min,
            optimizer=args.optimizer,
            momentum=args.momentum,
            scheduler=args.scheduler,
        )
    print(args)

    train_workers = count_available_workers(
        args.num_workers,
        bytes_per_worker=estimate_worker_shm_bytes(args.batch_size, args.image_size))
    eval_batch_size = args.eval_batch_size or args.batch_size
    val_workers = count_available_workers(
        min(2, args.num_workers),
        bytes_per_worker=estimate_worker_shm_bytes(eval_batch_size, args.image_size))
    # drop_last=False on validation: with drop_last=True a val set smaller than one
    # batch produced zero val batches, so val_loss never existed and
    # ModelCheckpoint(monitor="val_loss") silently saved nothing.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False,
        num_workers=train_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=train_workers > 0)
    val_loader = DataLoader(
        validation_dataset, batch_size=eval_batch_size, sampler=val_sampler, shuffle=False,
        num_workers=val_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=val_workers > 0)

    print("\nModel and dataset ready for training!")

    accelerator, devices = resolve_accelerator(args.accelerator)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=-1,
            every_n_epochs=1,
            dirpath=args.checkpoint_path,
            filename="googlenet_epoch_{epoch:02d}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=max(10, args.max_epochs // 5),
            mode="min",
            divergence_threshold=50.0,
        ),
    ]
    if args.swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=args.swa_lrs,
                swa_epoch_start=args.swa_epoch_start,
                annealing_epochs=args.annealing_epochs,
            )
        )

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
        callbacks=callbacks,
    )

    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=checkpoint_path)

    save_model_checkpoint(lightning_model, args.checkpoint_path, filename="googlenet_final.ckpt")
    return trainer


if __name__ == "__main__":
    main()
