#!/usr/bin/env python3
"""Does AlexNet's SGD failure come from ONE global lr across a 1.5e4 gradient spread?

Why this probe exists
---------------------
P2-25 asks whether unclipped SGD beats chance on the 64-class task.  Every
SGD arm run so far -- E031's 21 epochs and this session's 9-arm
clip x lr x precision grid plus two 7680-step long arms -- sat exactly on
chance (val_loss ~= ln(64) = 4.159, train_acc ~= 1.3-1.9 % against a
1.56 % chance rate), while Adam at the *same* budget left chance.

Ruled out by measurement, not by argument:

  * gradient clipping   -- clip off and clip=30 are also exactly at chance
  * the std=0.01 init   -- the paper init MEMORISES a fixed 32-image batch
                           with unclipped SGD (check_sgd_vs_init.py)
  * step budget         -- the Adam control converges at the same budget
  * the lr schedule     -- check_lr_schedule.py shows the cosine is stepped
                           per EPOCH and the lr stays healthy (this refuted
                           the "T_max is in epochs but stepped in steps"
                           theory, which looked plausible in the source)
  * dropout             -- --dropout off replaces every nn.Dropout with
                           nn.Identity and is still exactly at chance
  * weight decay        -- --weight_decay 0 is at chance too, alone and
                           combined with --dropout off.  The four arms
                           spanning that 2x2 cross-product move val_loss by
                           0.0007, i.e. less than one validation image.

What is left is the per-tensor gradient spread measured by
check_grad_scale.py: at init, AlexNet's gradient norms span a factor of
1.5e4 (conv1.bias 1.4e-3 ... classifier.6.weight 2.1e+1) while GoogLeNet's
are balanced.  Under a single global lr, the classifier moves ~1.5e4 times
faster than the feature extractor.  That predicts the classifier saturates
on the noise of frozen random conv features while conv1..conv5 never move.
Adam's per-parameter RMS normalisation is exactly the mechanism that
absorbs such a spread, which is why the Adam arm works.

A positive prediction follows: if the spread is the cause, then giving the
feature extractor a proportionally LARGER lr than the classifier should let
plain SGD escape chance -- no adaptive optimiser, no clipping, no init
change.  This probe tests that prediction directly.  If it holds, P2-25 has
a passing SGD configuration; if it fails, the spread is not the cause
either and P2-25 must stay open.

Each arm is a real training run on the cached subset (fresh draws per step,
not a fixed batch), evaluated on the cached validation split.

Usage (inside the container)::

    python probes/check_sgd_lr_groups.py                      # sweep the group ratio
    python probes/check_sgd_lr_groups.py --lr 0.1 --ratio 30  # one arm
    python probes/check_sgd_lr_groups.py --ratio 1 --dropout off        # no dropout
    python probes/check_sgd_lr_groups.py --ratio 1 --weight_decay 0     # no weight decay
    python probes/check_sgd_lr_groups.py --ratio 1 --weight_decay 0 --dropout off

Note that the probe exits non-zero when no arm escapes chance (VERDICT:
INCONCLUSIVE), so chain arms with ';' rather than '&&'.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "AlexNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FEATURE_PREFIXES = ("features.",)
CLASSIFIER_PREFIXES = ("classifier.",)


def disable_dropout(net) -> int:
    """Replace every nn.Dropout with nn.Identity; return how many were replaced."""
    import torch.nn as nn_mod
    n = 0
    for name, mod in list(net.named_children()):
        if isinstance(mod, nn_mod.Dropout):
            setattr(net, name, nn_mod.Identity())
            n += 1
        elif isinstance(mod, nn_mod.Sequential):
            for i, child in list(enumerate(mod)):
                if isinstance(child, nn_mod.Dropout):
                    mod[i] = nn_mod.Identity()
                    n += 1
    return n


def split_params(net, feature_lr: float, classifier_lr: float):
    """Return param groups: feature extractor at feature_lr, head at classifier_lr."""
    feats, head = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(FEATURE_PREFIXES):
            feats.append(p)
        elif name.startswith(CLASSIFIER_PREFIXES):
            head.append(p)
        else:  # any future top-level parameter goes with the head
            head.append(p)
    return [{"params": feats, "lr": feature_lr},
            {"params": head, "lr": classifier_lr}]


def evaluate(net, loader, device):
    import torch
    import torch.nn.functional as F
    net.eval()
    tot_l = tot_n = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = net(x)
            loss = F.cross_entropy(out, y, reduction="sum")
            tot_l += float(loss)
            tot_n += y.numel()
            correct += int((out.argmax(1) == y).sum())
    net.train()
    return tot_l / max(1, tot_n), correct / max(1, tot_n)


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
    ap.add_argument("--steps", type=int, default=256, help="optimizer steps per epoch")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1, help="classifier (head) lr")
    ap.add_argument("--ratio", default="1,3,10,30,100",
                    help="comma-separated feature-lr / head-lr ratios to sweep")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--dropout", default="on", choices=["on", "off"],
                    help="replace AlexNet's nn.Dropout modules with Identity. E035 shows "
                         "SGD MEMORISES a fixed batch while E033 shows it cannot fit a "
                         "stream of fresh batches (train_acc never exceeds 1.95%). The only "
                         "things that differ between those two settings are fresh draws per "
                         "step, dropout p=0.5 and weight_decay -- so they are the remaining "
                         "suspects and this is the arm that tests them.")
    ap.add_argument("--gradient_clip_val", default="1.0")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import model as alexnet_model
    from utils.datasets_utils import (
        load_dataset_subset, create_subset_sampler, create_eval_sampler)
    from utils.runtime import (
        seed_everything, count_available_workers, estimate_worker_shm_bytes)
    from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset

    args = parse_args()
    clip = None if str(args.gradient_clip_val).lower() in ("none", "off", "0") \
        else float(args.gradient_clip_val)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but unavailable; run probes/check_hardware.py")

    seed_everything(args.seed, deterministic=False)

    train_ds, val_ds, _ = load_dataset_subset(
        dataset_name=args.dataset_name, cache_root=args.cache_root,
        train_samples=args.train_samples, val_samples=args.val_samples,
        seed=args.seed, shuffle_buffer_size=args.shuffle_buffer)
    train_sampler, target_classes = create_subset_sampler(
        train_ds, max_num_classes=args.num_classes, target_classes=None,
        seed=args.seed, val_dataset=val_ds, num_samples=args.train_samples_per_epoch)
    target_classes = sorted(target_classes)
    K = len(target_classes)
    val_sampler = create_eval_sampler(val_ds, target_classes)

    train_ds = ImageNetSubsetDataset(train_ds, image_size=args.image_size,
                                     is_training=True, target_classes=target_classes)
    val_ds = ImageNetSubsetDataset(val_ds, image_size=args.image_size,
                                   is_training=False, target_classes=target_classes)
    w = count_available_workers(
        args.num_workers,
        bytes_per_worker=estimate_worker_shm_bytes(args.batch_size, args.image_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=max(1, w), pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, sampler=val_sampler,
                            num_workers=2, pin_memory=True, drop_last=False,
                            persistent_workers=True)

    chance = math.log(K)
    ratios = [float(x) for x in str(args.ratio).split(",") if x.strip()]
    print(f"torch {torch.__version__}  device={device}  classes={K}  "
          f"image_size={args.image_size}  batch={args.batch_size}")
    print(f"steps={args.steps} epochs={args.epochs} head_lr={args.lr:g} "
          f"clip={clip} momentum={args.momentum} wd={args.weight_decay} seed={args.seed}")
    print(f"chance loss ln({K}) = {chance:.4f}; PASS rule (as in check_learn.py) = "
          f"val_loss < {0.6 * chance:.4f}")
    print(f"arms: feature_lr / head_lr in {ratios}   dropout={args.dropout}")

    rows = []
    for ratio in ratios:
        seed_everything(args.seed, deterministic=False)
        net = alexnet_model.AlexNet(num_classes=K, image_size=args.image_size).to(device)
        n_drop = disable_dropout(net) if args.dropout == "off" else 0
        net.train()
        groups = split_params(net, feature_lr=args.lr * ratio, classifier_lr=args.lr)
        n_feat = sum(p.numel() for p in groups[0]["params"])
        n_head = sum(p.numel() for p in groups[1]["params"])
        # Snapshot init so we can report how far each group actually travelled.
        f0 = [p.detach().clone() for p in groups[0]["params"]]
        h0 = [p.detach().clone() for p in groups[1]["params"]]
        opt = torch.optim.SGD(groups, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        it = iter(train_loader)
        first = last = None
        nan_at = None
        t0 = time.time()
        for step in range(1, args.steps * args.epochs + 1):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss = F.cross_entropy(out, y)
            if not torch.isfinite(loss):
                nan_at = step
                break
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            if first is None:
                first = float(loss.detach())
            last = float(loss.detach())
        vl, va = evaluate(net, val_loader, device)
        with torch.no_grad():
            fmove = max(float((p - q).norm()) for p, q in zip(groups[0]["params"], f0))
            hmove = max(float((p - q).norm()) for p, q in zip(groups[1]["params"], h0))
        rows.append((ratio, first, last, vl, va, fmove, hmove, nan_at))
        print(f"ratio={ratio:>6g}  feat_lr={args.lr * ratio:<10.4g} head_lr={args.lr:<8g} "
              f"dropout={'off' if args.dropout == 'off' else 'on'}({n_drop}) "
              f"loss {first:.4f}->{last:.4f}  val_loss={vl:.4f} val_acc={va * 100:5.2f}%  "
              f"max|dw| feat={fmove:.3e} head={hmove:.3e}"
              + (f"  NaN@{nan_at}" if nan_at else "")
              + f"  [{time.time() - t0:.0f}s]")

    print("\n=== summary (chance val_loss = %.4f, chance val_acc = %.2f %%) ===" %
          (chance, 100.0 / K))
    print(f"{'ratio':>7} {'feat_lr':>10} {'val_loss':>9} {'val_acc%':>9} "
          f"{'feat |dw|':>11} {'head |dw|':>11} {'verdict':>8}")
    winners = []
    for ratio, first, last, vl, va, fmove, hmove, nan_at in rows:
        ok = (nan_at is None) and (vl < 0.6 * chance)
        if ok:
            winners.append(ratio)
        print(f"{ratio:>7g} {args.lr * ratio:>10.4g} {vl:>9.4f} {va * 100:>9.2f} "
              f"{fmove:>11.3e} {hmove:>11.3e} {'PASS' if ok else 'chance':>8}")

    print(f"\ntrainable params: features={n_feat:,} head={n_head:,}")
    if winners:
        print(f"\nVERDICT: PASS -- plain SGD beats chance at feature/head lr ratio(s) "
              f"{winners}.\nThe 1.5e4 init gradient spread under a single global lr was "
              f"the blocker;\nno adaptive optimiser and no clipping is required.")
        return 0
    print("\nVERDICT: INCONCLUSIVE -- no ratio escapes chance, so the gradient spread is "
          "not\n         the (only) blocker either. P2-25 stays open; do not close it on "
          "this result.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
