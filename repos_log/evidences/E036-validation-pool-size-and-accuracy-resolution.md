# E036 — Validation is 153 images (0.654 % resolution); `val_samples=16384` gives 1030 (0.0971 %)

| | |
|---|---|
| **Status** | CONFIRMED — `val_samples=16384` yields **1030** in-class validation images, resolution **0.0971 %**; `P2-27` acceptance met |
| **Cards** | `P2-27` (resolved), affects every `val_acc` figure in the repo |
| **Command** | `docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_val_pool.py --val_samples 16384 --predict 1000'` |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists

`create_eval_sampler` covers every in-class validation sample **exactly once**. So `val_acc` is
not a continuous measurement — it is a multiple of `100 / (in-class val images)`. On the canonical
`val_samples=2048` cache, `choose_target_classes` keeps 64 classes that together hold only **153**
validation images. Every `val_acc` this repo has ever reported is therefore a multiple of
`100/153 = 0.6536 %`.

That has three consequences the card lists, all of which are visible in the existing evidence:

1. A genuine 2–3 % improvement is **invisible** — it is 3–5 images.
2. A single lucky epoch can look like a breakthrough.
3. The epoch-to-epoch "oscillation" HANDOFF_003 read as training instability is 1–7 images
   changing side.

The in-class count is **not** `val_samples × 64/1000`. `choose_target_classes` prefers classes that
have validation coverage, so the in-class fraction is much higher than the naive estimate — and it
is a property of the cache, so it has to be measured, not computed. That is what this probe does.

## Command A — measure the cache the deliverable shipped with

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_val_pool.py --val_samples 2048'
```

### Raw output, verbatim

```
Loading cached subset of ILSVRC/imagenet-1k from /workspace/Data/datasets/ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538
Train dataset size: 51200
Validation dataset size: 2048
Selected 64 classes; per-class cached counts: min=62, median=65, max=73
cache loaded in 0.0s  train=51200  val=2048
num_classes=64  seed=42  classes chosen=64
IN-CLASS TRAIN = 4153  (8.1113% of the cache)
IN-CLASS VAL   = 153  (7.4707% of the cache)
classes with >=1 val image: 64/64
per-class val counts (in-class): min=1 p25=1 median=2 p75=3 max=6

val_acc resolution = 100/153 = 0.6536 %  (one image = 0.6536 % of accuracy)
smallest detectable change at 1 sigma ~ 0.6536 %; a 2 % effect is 3.1 images

VERDICT: FAIL -- only 153 in-class validation images; P2-27 needs >= 1000.
         Raise --val_samples (see predict above).
```

This reproduces the card's number exactly. Note `min=1`: at least one of the 64 classes is scored
on a **single** image, so that class's per-class accuracy is either 0 % or 100 %.

## Command B — measure the enlarged pool and derive the documented value

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_val_pool.py --val_samples 16384 --predict 1000'
```

### Raw output, verbatim

```
Loading cached subset of ILSVRC/imagenet-1k from /workspace/Data/datasets/ILSVRC__imagenet-1k_51200t_16384v_auto_cf2d39fe2ecf
Train dataset size: 51200
Validation dataset size: 16384
Selected 64 classes; per-class cached counts: min=62, median=65, max=73
cache loaded in 0.1s  train=51200  val=16384
num_classes=64  seed=42  classes chosen=64
IN-CLASS TRAIN = 4195  (8.1934% of the cache)
IN-CLASS VAL   = 1030  (6.2866% of the cache)
classes with >=1 val image: 64/64
per-class val counts (in-class): min=9 p25=14 median=16 p75=18 max=24

val_acc resolution = 100/1030 = 0.0971 %  (one image = 0.0971 % of accuracy)
smallest detectable change at 1 sigma ~ 0.0971 %; a 2 % effect is 20.6 images

predict: in-class fraction = 6.2866% -> val_samples >= 16384 for >= 1000 in-class val images
         (that pool's val_acc resolution would be <= 0.0971 %)

VERDICT: PASS -- 1030 in-class validation images (>= 1000 required by P2-27).
```

**`val_samples=16384` is the documented answer.** 1030 in-class images, 0.0971 % resolution — a
6.7× improvement, and a 2 % effect is now 20.6 images rather than 3.1. The worst class goes from
1 image to 9.

## The caveat that travels with the fix

`val_samples` is not a neutral knob. It feeds `choose_target_classes`'s `min_val_count` filter, so
changing it changes **which 64 classes are chosen**. Measured directly:

```
2048  n= 64  sum= 28254  first8= [39, 46, 54, 56, 81, 106, 112, 121]
16384 n= 64  sum= 29783  first8= [39, 46, 48, 54, 56, 81, 111, 112]
   identical: False | only in 2048:  [106, 170, 272, 275, 381, 417, 548, 637]
                    | only in 16384: [48, 111, 280, 538, 639, 851, 893, 975]
```

**8 of the 64 classes differ.** Two consequences:

- Accuracy numbers from the 2048 caches ([E030](./E030-googlenet-converged-run.md),
  [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md),
  [E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)) are **not comparable** with
  runs on the 16384 pool. Re-baseline; do not compare across.
- `IN-CLASS TRAIN` also moves (4153 → 4195), so the effective training distribution changes too.

This is why E033 deliberately kept `--val_samples 2048`: it must stay comparable with E031.

## Changes made

- `AlexNet/train_final.py` `CONFIG["val_samples"]`: `2048 → 16384`, with the resolution and the
  class-drift caveat in the adjacent comment.
- `GoogLeNet/train_final.py` `CONFIG["val_samples"]`: same change.
- `goal.md`: the "`val_acc` rests on 153 validation images" clause replaced with the measured
  resolution and the documented value.
- `SMOKE_OVERRIDES["val_samples"]` stays at 256 — a smoke test only has to execute.

## Caveats

- The 16384 cache is **7.6 GB** and is a *different* cache directory: `_cache_key` includes
  `val_samples`, so this is a new download, not a re-slice. Disk was at 90 % (95 G free) when it
  was built; prune old caches.
- `shuffle_buffer_size` is excluded from `_cache_key`/`identity_keys`, so changing it does **not**
  invalidate a cache — do not expect a new directory from it.
- The in-class fraction (6.29 %) is specific to `seed=42`, `train_samples=51200` and
  `num_classes=64`. A different `num_classes` needs a re-measure; `--predict` recomputes it.
- Resolution is a *quantisation* statement, not a statistical-power statement. With ~16 images per
  class, per-class accuracies are still noisy; only the aggregate is at 0.0971 % granularity.

## Verdict

`P2-27`'s acceptance criterion — "a documented `val_samples` value that gives ≥1 000 in-class
validation images, the resulting resolution stated in `goal.md`, and the `CONFIG` dicts updated" —
is met by **`val_samples=16384` → 1030 in-class images → 0.0971 % resolution**.
