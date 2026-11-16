# E029 — Augmentation A/B: the train arm memorises at 227 px, so `P2-15` is not the convergence blocker

| | |
|---|---|
| **Status** | **SUPERSEDED by [E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md)** — the A/B in this file is invalid: `build_batch()` was called once per arm, so the eval and train arms memorised *two different* 32-image batches, and the batch was drawn from the unseeded global RNG so `--seed` did not pin it. The 4/4 PASS below did not reproduce once both bugs were fixed. Kept per the evidence contract (superseded evidence stays). The `run_cell()` seeding-order bug and the `median`/`max` addition described here **are** valid and still stand. |
| **Cards** | `P2-15` (resolved for 227 px), `P2-21` (extends it to the train arm) |
| **Command** | `python probes/check_learn.py --steps 400 --arm both --image-size 227` and `... --image-size 160` |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev5` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_005](../handoffs/HANDOFF_005.md) |

## Correction (HANDOFF_005, same session)

Everything under **Output** below was produced by a probe with two RNG defects, found later in the
same session while re-running this file's own documented command as a regression check:

1. `main()` did not seed before `build_batch()`, and `build_batch()` calls `list(sampler)` on a
   `WeightedRandomSampler`, which draws from the **global torch RNG**. The "fixed" batch was
   re-drawn on every invocation, so `--seed` did not pin it.
2. `build_batch()` was called **inside the arm loop**, so each arm re-drew the sampler and the
   eval/train A/B compared two different batches — the exact confound `--arm` exists to remove.

Re-running the identical command line with both fixed gives **2/4 PASS** at 227 px, not 4/4. See
[E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md), which also shows the probe's verdict is
dominated by its own default `lr=3e-3`. `P2-15` is reopened.

## Why this experiment, and why it had not been run

`P2-15` claimed `Resize(256)` → `RandomCrop(image_size)` was destroying signal. The evidence for
it was HANDOFF_003's observation that both transform arms failed to memorise 89 real images while
synthetic noise memorised fine. [E023](./E023-learning-matrix-both-devices-both-caches.md) then
showed the **eval** arm memorises comfortably, and narrowed the card to:

> **Still missing:** the train arm has never actually been tested in isolation — it is still a
> hypothesis, not a confirmed cause.

`probes/check_learn.py` built its batch with `is_training=False` and a fixed tensor, so it could
only ever test the eval arm. This experiment adds `--arm {eval,train,both}` (plus the `--is_training`
shorthand the card asked for). The train arm keeps the **same decoded PIL images** and re-applies the
real training transform (`Resize(256) → RandomCrop → RandomHorizontalFlip → ColorJitter →
ToTensor → Normalize`) at every step, so the A/B differs *only* in the transform — not in which
images were chosen, which is what makes it an A/B at all.

## A probe bug had to be fixed first, or the result was noise

`run_cell()` called `torch.manual_seed(seed)` **after** `make()` constructed the network, so the
initial weights came from whatever RNG state the previous cell left behind. The same command gave
`PASS` and `FAIL` on different invocations of the same cell. Reproduced before fixing:

```
=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=eval  (32 images, 10 classes, (227, 227), seed=42) ===
  AlexNet    cuda  eval   chance 2.3026   2.3064 -> 3.4102   acc 0.969    38.9s   FAIL
```

`loss 3.41` with `acc 0.969` is not a contradiction — it is a 32-image batch where 31 samples are
memorised and one has a very large per-sample loss, and the *mean* is what the pass rule reads. The
fix is twofold: seed **before** `make()`, and print `median`/`max` per-sample loss next to the mean
so an outlier-dominated mean is visible instead of mysterious. A second run of the identical
unseeded cell gave `last 0.0022 acc 1.000 PASS`, which is what proved the instability was the
unseeded init and not the data.

## Output — `--image-size 227` (the value both `train_final.py` CONFIGs use)

```
torch 2.10.0+cu128   cuda available: True   gpu: NVIDIA RTX A1000 6GB Laptop GPU
devices: ['cuda']
caches:  ['ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538']
Selected 10 classes; per-class cached counts: min=67, median=68, max=73

=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=eval  (32 images, 10 classes, (227, 227), seed=42) ===
  AlexNet    cuda  eval   chance 2.3026   3.1981 -> 0.1997   med -0.0000 max 4.094   acc 0.906    38.7s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6853 -> 0.0001   med -0.0000 max 0.002   acc 1.000    46.8s   PASS
Selected 10 classes; per-class cached counts: min=67, median=68, max=73

=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=train  (32 images, 10 classes, (227, 227), seed=42) ===
  AlexNet    cuda  train  chance 2.3026   2.7073 -> 0.1803   med 0.0000 max 2.138   acc 0.906    82.1s   PASS
  GoogLeNet  cuda  train  chance 3.6841   3.6829 -> 0.3799   med 0.0001 max 7.082   acc 0.938    91.6s   PASS

============================================================================================
model      device  arm    cache                                   result
--------------------------------------------------------------------------------------------
AlexNet    cuda    eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
GoogLeNet  cuda    eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
AlexNet    cuda    train  ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
GoogLeNet  cuda    train  ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
============================================================================================

OVERALL: all 4 cells learned -- both networks learn from real images on every device and arm tested, on every cached subset.
```

**At the resolution the deliverable actually trains at, the augmentation arm memorises 32 real
images to 90–94 % batch accuracy on both networks.** The `P2-15` hypothesis does not reproduce at
227 px.

## Output — `--image-size 160` (both caches)

```
=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=train  (32 images, 10 classes, (160, 160), seed=42) ===
  AlexNet    cuda  train  chance 2.3026   2.4813 -> 1.1187   med 0.5517 max 6.883   acc 0.625    73.8s   PASS
  GoogLeNet  cuda  train  chance 3.6841   3.6858 -> 0.0296   med 0.0001 max 0.162   acc 1.000    88.6s   PASS
Selected 10 classes; per-class cached counts: min=3, median=4, max=5

=== ILSVRC__imagenet-1k_1024t_256v_auto_bdedb67a1a9e  arm=eval  (32 images, 10 classes, (160, 160), seed=42) ===
  AlexNet    cuda  eval   chance 2.3026   2.5347 -> 0.0000   med -0.0000 max -0.000   acc 1.000    19.2s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6882 -> 0.6239   med -0.0000 max 6.273   acc 0.969    27.3s   PASS
Selected 10 classes; per-class cached counts: min=3, median=4, max=5

=== ILSVRC__imagenet-1k_1024t_256v_auto_bdedb67a1a9e  arm=train  (32 images, 10 classes, (160, 160), seed=42) ===
  AlexNet    cuda  train  chance 2.3026   2.4352 -> 2.2521   med 2.3518 max 3.130   acc 0.156    72.6s   FAIL
  GoogLeNet  cuda  train  chance 3.6841   3.6885 -> 0.4598   med 0.0002 max 7.081   acc 0.969    81.6s   PASS

============================================================================================
model      device  arm    cache                                   result
--------------------------------------------------------------------------------------------
AlexNet    cuda    eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
GoogLeNet  cuda    eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
AlexNet    cuda    train  ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
GoogLeNet  cuda    train  ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
AlexNet    cuda    eval   ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS
GoogLeNet  cuda    eval   ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS
AlexNet    cuda    train  ILSVRC__imagenet-1k_1024t_256v_auto_bde FAIL
GoogLeNet  cuda    train  ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS
============================================================================================

OVERALL: 1/8 cell(s) did not beat chance: AlexNet/cuda/train/ILSVRC__imagenet-1k_1024t_256v_auto_bdedb67a1a9e
```

## What this settles

1. **`P2-15` is not the reason neither network has converged.** At `image_size=227` — the value in
   both `train_final.py` `CONFIG` dicts — the train arm memorises as well as the eval arm. Whatever
   is stopping convergence is not `RandomCrop`/`ColorJitter`.
2. **Augmentation is nonetheless a measurable drag on AlexNet, and it is size-dependent.** At 160 px
   on the sparse 1024-sample cache, eval reaches `acc 1.000` / loss `0.0000` while train reaches
   `acc 0.156` / loss `2.2521` on the *same 32 images*. That is the signature HANDOFF_003 described,
   and it is real — but it appears at 160 px, not at the 227 px the recipe uses. `Resize(256)` +
   `RandomCrop(160)` keeps `(160/256)² = 39 %` of the area; at 227 px it keeps `79 %`.
3. **GoogLeNet is augmentation-robust at both sizes** (train arm ≥ 0.938 accuracy everywhere), so the
   drag is AlexNet-specific — consistent with its much larger, more aggressively initialised
   classifier and its lack of BatchNorm.

## What this does **not** show

- It does **not** show the augmentation is *optimal*, only that it does not prevent learning. The
  160 px AlexNet result argues the crop is harsher than it needs to be below ~200 px; `P3-7`
  (paper-faithful `Resize(256)`→`RandomCrop(224)` with padding) remains open and is the natural fix.
- It is still memorisation of 32 images, not convergence over a dataset. That is
  [E030](./E030-googlenet-converged-run.md).
- Only `cuda` was swept here. E023 already covers CPU for the eval arm; the train arm on CPU was not
  re-run because it is ~12–18× slower for the same verdict.
- `ColorJitter` and `RandomHorizontalFlip` were not ablated individually. Isolating which of the
  three train-arm ops costs AlexNet the 160 px cell is a follow-up, not a claim made here.
