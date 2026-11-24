#!/usr/bin/env python3
"""Record the ACTUAL learning rate the optimiser sees during a real run.

Why this probe exists
---------------------
P2-25 asks whether unclipped SGD beats chance on the 64-class task.  Every
SGD arm ever run in this repo -- E031's 21 epochs, this session's 9-arm
clip x lr x precision grid -- sat exactly on chance (val_loss ~= ln(64) =
4.159), while Adam at the identical budget left chance.  The three usual
explanations were each tested and each ruled out:

  * gradient clipping  -- clip off and clip=30 are also at chance
                          (this session's grid, probes/check_grad_scale.py)
  * the uniform std=0.01 init -- the paper init MEMORISES a fixed batch with
                          unclipped SGD (probes/check_sgd_vs_init.py)
  * too few steps      -- the Adam control converges at the same budget

That leaves the learning-rate *schedule*.  ``AlexNet/model.py::
configure_optimizers`` builds

    CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

and returns it as a bare ``{"optimizer": ..., "lr_scheduler": ...}`` entry.
Lightning's documented default for that form is ``interval="step"``,
``frequency=1`` -- the scheduler is advanced once per OPTIMISER step, not
once per epoch.  ``T_max`` is therefore being *interpreted* in steps while
it is being *supplied* in epochs.

This probe does not argue from the source.  It builds the loaders exactly
the way ``AlexNet/train.py`` does, attaches a callback that samples
``optimizer.param_groups[0]["lr"]`` at every optimizer step, and prints the
trajectory -- so the claim is directly falsifiable by running it.

Read-only: no weights are written.

Usage (inside the container)::

    python probes/check_lr_schedule.py --optimizer sgd --lr 0.01 --scheduler cosine
    python probes/check_lr_schedule.py --optimizer sgd --lr 0.01 --scheduler none
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "AlexNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_name", default="ILSVRC/imagenet-1k")
    ap.add_argument("--cache_root", default=os.path.join(PROJECT_ROOT, "Data", "datasets"))
    ap.add_argument("--num_classes", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=227)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_samples", type=int, default=51200)
    ap.add_argument("--train_samples_per_epoch", type=int, default=16384)
    ap.add_argument("--val_samples", type=int, default=2048)
    ap.add_argument("--shuffle_buffer", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=None,
                    help="limit_train_batches (default: a full epoch)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    ap.add_argument("--lr_step_size", type=int, default=30)
    ap.add_argument("--lr_gamma", type=float, default=0.1)
    ap.add_argument("--gradient_clip_val", default="1.0")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    import lightning as pl
    from lightning.pytorch.callbacks import Callback

    import model as alexnet_model
    from utils.datasets_utils import (
        load_dataset_subset, create_subset_sampler, create_eval_sampler)
    from utils.runtime import (
        seed_everything, count_available_workers, estimate_worker_shm_bytes)
    from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset

    args = parse_args()
    clip = None if str(args.gradient_clip_val).lower() in ("none", "off", "0") \
        else float(args.gradient_clip_val)
    seed_everything(args.seed, deterministic=False)

    train_dataset, validation_dataset, _ = load_dataset_subset(
        dataset_name=args.dataset_name, cache_root=args.cache_root,
        train_samples=args.train_samples, val_samples=args.val_samples,
        seed=args.seed, shuffle_buffer_size=args.shuffle_buffer)

    train_sampler, target_classes = create_subset_sampler(
        train_dataset, max_num_classes=args.num_classes, target_classes=None,
        seed=args.seed, val_dataset=validation_dataset,
        num_samples=args.train_samples_per_epoch)
    target_classes = sorted(target_classes)
    K = len(target_classes)
    val_sampler = create_eval_sampler(validation_dataset, target_classes)

    train_dataset = ImageNetSubsetDataset(
        train_dataset, image_size=args.image_size, is_training=True,
        target_classes=target_classes)
    validation_dataset = ImageNetSubsetDataset(
        validation_dataset, image_size=args.image_size, is_training=False,
        target_classes=target_classes)

    model = alexnet_model.AlexNetLightning(
        num_classes=K, target_classes=target_classes, image_size=args.image_size,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma,
        log_grad_norms=False, optimizer=args.optimizer, scheduler=args.scheduler)

    workers = count_available_workers(
        args.num_workers,
        bytes_per_worker=estimate_worker_shm_bytes(args.batch_size, args.image_size))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, persistent_workers=workers > 0)

    class LRSampler(Callback):
        def __init__(self) -> None:
            self.lrs: list[float] = []

        def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
            self.lrs.append(float(optimizer.param_groups[0]["lr"]))

    sampler = LRSampler()
    trainer = pl.Trainer(
        accelerator=args.device, devices=1, precision=args.precision,
        max_epochs=args.epochs, limit_train_batches=args.steps,
        limit_val_batches=0, num_sanity_val_steps=0,
        gradient_clip_val=clip, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, callbacks=[sampler])
    trainer.fit(model, train_dataloaders=train_loader)

    lrs = sampler.lrs
    nominal = float(args.lr)
    steps_per_epoch = len(train_loader)
    print("\n=== lr trajectory ===")
    print(f"  optimizer={args.optimizer}  scheduler={args.scheduler}  nominal lr={nominal:g}")
    print(f"  max_epochs={args.epochs}  steps/epoch={steps_per_epoch}  classes={K}  "
          f"image_size={args.image_size}")
    if not lrs:
        print("FAIL  no optimizer steps were recorded")
        return 1
    print(f"  optimizer steps recorded : {len(lrs)}")
    print(f"  max / min / final lr     : {max(lrs):.6e} / {min(lrs):.6e} / {lrs[-1]:.6e}")
    print("  sampled trajectory:")
    idx = sorted(set(list(range(0, len(lrs), max(1, len(lrs) // 12))) + [len(lrs) - 1]))
    for i in idx:
        print(f"    step {i:>6}: lr={lrs[i]:.6e}  ({lrs[i] / nominal * 100:6.2f} % of nominal)")

    dead = sum(1 for x in lrs if x < 0.01 * nominal)
    frac_dead = dead / len(lrs)
    mean_lr = sum(lrs) / len(lrs)
    print(f"\n  steps below 1 % of nominal lr : {dead}/{len(lrs)} ({frac_dead:.1%})")
    print(f"  mean lr over the run          : {mean_lr:.6e} ({mean_lr / nominal:.1%} of nominal)")

    if args.scheduler == "cosine" and frac_dead > 0.5:
        print("\nVERDICT: FAIL -- CosineAnnealingLR is advanced once per OPTIMISER STEP "
              "while\n         T_max is supplied in epochs, so the run spends most of its "
              "time at lr~0.\n         Fix: return the scheduler with interval='epoch', or "
              "set T_max to the total number of optimizer steps.")
        return 1
    print("\nVERDICT: PASS -- the observed lr stays in a usable range for this configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
