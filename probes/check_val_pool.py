#!/usr/bin/env python3
"""How many IN-CLASS validation images does a given ``val_samples`` actually give us?

Why this probe exists
---------------------
``P2-27``: ``create_eval_sampler`` covers every in-class validation sample
exactly once, so ``val_acc`` is a multiple of ``100 / (in-class val images)``.
The canonical ``val_samples=2048`` cache leaves only **153** images inside the
64 chosen classes -- a resolution of 0.654 %.  Every ``val_acc`` in this repo
is therefore quantised in 0.654 % steps, a real 2-3 % improvement is
invisible, and the epoch-to-epoch "oscillation" HANDOFF_003 read as training
instability is 1-7 images changing side.

The card asks for a documented ``val_samples`` that yields >= 1000 in-class
validation images, with the resulting resolution stated in ``goal.md``.  The
in-class count is *not* ``val_samples * 64/1000`` -- ``choose_target_classes``
prefers classes that have validation coverage, so the in-class fraction is
much higher than the naive estimate, and it is a property of the cache.  It
has to be measured, not computed.

This probe measures it for a cache and reports the resolution.  With
``--predict`` it also converts the measured fraction into the ``val_samples``
needed to reach a target in-class count, so the documented value is derived
from a measurement rather than guessed.

Usage (inside the container)::

    # measure the cache the deliverable ships with
    python probes/check_val_pool.py --val_samples 2048

    # measure the enlarged pool and derive the documented value
    python probes/check_val_pool.py --val_samples 16384 --predict 1000

Both invocations reuse an existing cache if one matches the key; building the
16384-image pool from scratch takes several minutes and ~7.6 GB.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.datasets_utils import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_CACHE_ROOT,
    DEFAULT_SHUFFLE_BUFFER,
    choose_target_classes,
    label_distribution,
    load_dataset_subset,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_name", default=DEFAULT_DATASET)
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--train_samples", type=int, default=51200)
    ap.add_argument("--val_samples", type=int, default=2048)
    ap.add_argument("--num_classes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the seed baked into train_final.py CONFIG; it is "
                         "part of the cache key")
    ap.add_argument("--shuffle_buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    ap.add_argument("--predict", type=int, default=None, metavar="N",
                    help="also report the val_samples needed for >= N in-class val images")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    train, val, _ = load_dataset_subset(
        dataset_name=args.dataset_name, cache_root=args.cache_root,
        train_samples=args.train_samples, val_samples=args.val_samples,
        seed=args.seed, shuffle_buffer_size=args.shuffle_buffer)
    build_s = time.time() - t0

    target_classes = choose_target_classes(
        train, num_classes=args.num_classes, seed=args.seed, val_dataset=val)
    ts = set(target_classes)

    tl = label_distribution(train)
    vl = label_distribution(val)
    in_train = sum(c for lbl, c in tl.items() if lbl in ts)
    in_val = sum(c for lbl, c in vl.items() if lbl in ts)
    val_classes_present = len([lbl for lbl in vl if lbl in ts])

    print(f"cache loaded in {build_s:.1f}s  train={len(train)}  val={len(val)}")
    print(f"num_classes={args.num_classes}  seed={args.seed}  "
          f"classes chosen={len(target_classes)}")
    print(f"IN-CLASS TRAIN = {in_train}  ({in_train / max(1, len(train)):.4%} of the cache)")
    print(f"IN-CLASS VAL   = {in_val}  ({in_val / max(1, len(val)):.4%} of the cache)")
    print(f"classes with >=1 val image: {val_classes_present}/{len(target_classes)}")

    qv = sorted(vl[lbl] for lbl in vl if lbl in ts)
    if qv:
        print("per-class val counts (in-class): "
              f"min={qv[0]} p25={qv[len(qv) // 4]} median={qv[len(qv) // 2]} "
              f"p75={qv[3 * len(qv) // 4]} max={qv[-1]}")

    if in_val == 0:
        print("\nVERDICT: FAIL -- no in-class validation images; val_acc is undefined.")
        return 1

    resolution = 100.0 / in_val
    print(f"\nval_acc resolution = 100/{in_val} = {resolution:.4f} %  "
          f"(one image = {resolution:.4f} % of accuracy)")
    print(f"smallest detectable change at 1 sigma ~ {resolution:.4f} %; "
          f"a 2 % effect is {2.0 / resolution:.1f} images")

    if args.predict:
        frac = in_val / max(1, len(val))
        need = int(args.predict / frac) + 1
        # Round up to a multiple of 1024 so the value stays cache-friendly.
        need = ((need + 1023) // 1024) * 1024
        print(f"\npredict: in-class fraction = {frac:.4%} -> val_samples >= {need} "
              f"for >= {args.predict} in-class val images")
        print(f"         (that pool's val_acc resolution would be "
              f"<= {100.0 / (need * frac):.4f} %)")

    if in_val >= 1000:
        print(f"\nVERDICT: PASS -- {in_val} in-class validation images (>= 1000 required "
              f"by P2-27).")
        return 0
    print(f"\nVERDICT: FAIL -- only {in_val} in-class validation images; P2-27 needs >= 1000. "
          f"\n         Raise --val_samples (see predict above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
