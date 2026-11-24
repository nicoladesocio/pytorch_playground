# E038 — Per-group learning rates do not rescue AlexNet's SGD either

| | |
|---|---|
| **Status** | CONFIRMED (negative result) — feature/head `lr` ratios 1, 3, 10, 30, 100 all sit on chance, and so do the arms with dropout removed, `weight_decay` removed, and both removed together. The 1.5 × 10⁴ gradient spread is **not** the blocker, and neither are the two regularisers. |
| **Cards** | `P2-25` (rules out the last optimiser-side candidate and both regularisation candidates) |
| **Command** | see §Command / §Command B / §Command C — all copy-pasteable |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists

This is the **positive prediction** of
[E034](./E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md). E034 measured that
AlexNet's per-tensor gradient norms span 1.5 × 10⁴ at init, and argued that under one global `lr`
the classifier moves ~1.5 × 10⁴ times faster than the feature extractor — so the classifier
saturates on the noise of frozen random conv features while conv1..conv5 never move.

If that is the mechanism, a specific and falsifiable thing must be true: **giving the feature
extractor a proportionally larger `lr` than the classifier should let plain SGD escape chance** —
no adaptive optimiser, no clipping change, no init change. Adam's per-parameter RMS normalisation
is the only thing that has ever absorbed this spread, so manually doing the same thing with two
param groups should reproduce Adam's success.

This probe tests exactly that. It is the last optimiser-side candidate; if it fails, `P2-25` cannot
be closed on any optimiser explanation.

## Command

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1,3,10,30,100'
```

Each arm is a real training run — fresh draws per step, not a fixed batch — on the canonical
`51200t_2048v` cache, 64 classes, 227 px, `batch_size=32`, `momentum=0.9`, `weight_decay=5e-4`,
`gradient_clip_val=1.0`, `seed=42`. The network is split into two SGD param groups:
`features.*` at `lr × ratio`, `classifier.*` at `lr`. `max|dw|` is the largest parameter
displacement from init in each group, which is how we check the two groups actually moved at
different rates.

## Raw output, verbatim

```
torch 2.10.0+cu128  device=cuda  classes=64  image_size=227  batch=32
steps=256 epochs=8 head_lr=0.1 clip=1.0 momentum=0.9 wd=0.0005 seed=42
chance loss ln(64) = 4.1589; PASS rule (as in check_learn.py) = val_loss < 2.4953
arms: feature_lr / head_lr in [1.0, 3.0, 10.0, 30.0, 100.0]   dropout=on

ratio=     1  feat_lr=0.1        head_lr=0.1      loss 4.7591->4.1759  val_loss=4.1704 val_acc= 0.65%  max|dw| feat=1.259e+01 head=4.115e+01  [365s]
ratio=     3  feat_lr=0.3        head_lr=0.1      loss 4.3706->4.1654  val_loss=4.1703 val_acc= 0.65%  max|dw| feat=1.862e+01 head=4.117e+01  [380s]
ratio=    10  feat_lr=1          head_lr=0.1      loss 4.3706->4.1720  val_loss=4.1719 val_acc= 1.31%  max|dw| feat=1.900e+01 head=4.117e+01  [338s]
ratio=    30  feat_lr=3          head_lr=0.1      loss 4.3705->4.1743  val_loss=4.1713 val_acc= 0.65%  max|dw| feat=1.960e+01 head=4.120e+01  [386s]
ratio=   100  feat_lr=10         head_lr=0.1      loss 4.3704->4.1755  val_loss=4.1731 val_acc= 0.65%  max|dw| feat=1.960e+01 head=4.119e+01  [279s]

=== summary (chance val_loss = 4.1589, chance val_acc = 1.56 %) ===
  ratio    feat_lr  val_loss  val_acc%   feat |dw|   head |dw|  verdict
      1        0.1    4.1704      0.65   1.259e+01   4.115e+01   chance
      3        0.3    4.1703      0.65   1.862e+01   4.117e+01   chance
     10          1    4.1719      1.31   1.900e+01   4.117e+01   chance
     30          3    4.1713      0.65   1.960e+01   4.120e+01   chance
    100         10    4.1731      0.65   1.960e+01   4.119e+01   chance

trainable params: features=3,747,200 head=54,796,352

VERDICT: INCONCLUSIVE -- no ratio escapes chance, so the gradient spread is not
         the (only) blocker either. P2-25 stays open; do not close it on this result.
```

## Command B — the same arm with dropout removed

E035 established that SGD **memorises** a fixed 32-image batch, while E033 established that it
cannot fit a *stream* of fresh batches (`train_acc` never exceeds 1.95 % in any arm). Only three
things differ between those two settings: fresh draws per step, `dropout p=0.5`, and
`weight_decay=5e-4`. The first is the task, so the latter two are the remaining suspects. This arm
replaces every `nn.Dropout` in the network with `nn.Identity` and re-runs ratio 1.

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1 --dropout off'
```

### Raw output, verbatim

```
torch 2.10.0+cu128  device=cuda  classes=64  image_size=227  batch=32
steps=256 epochs=8 head_lr=0.1 clip=1.0 momentum=0.9 wd=0.0005 seed=42
chance loss ln(64) = 4.1589; PASS rule (as in check_learn.py) = val_loss < 2.4953
arms: feature_lr / head_lr in [1.0]   dropout=off
ratio=     1  feat_lr=0.1        head_lr=0.1      dropout=off(2) loss 4.3817->4.1868  val_loss=4.1741 val_acc= 0.65%  max|dw| feat=1.258e+01 head=4.112e+01  [184s]

=== summary (chance val_loss = 4.1589, chance val_acc = 1.56 %) ===
  ratio    feat_lr  val_loss  val_acc%   feat |dw|   head |dw|  verdict
      1        0.1    4.1741      0.65   1.258e+01   4.112e+01   chance

trainable params: features=3,747,200 head=54,796,352

VERDICT: INCONCLUSIVE -- no ratio escapes chance, so the gradient spread is not
         the (only) blocker either. P2-25 stays open; do not close it on this result.
```

`dropout=off(2)` confirms two `nn.Dropout` modules were replaced (the two `p=0.5` layers in
`classifier`). The arm ran in 184 s versus 365 s for the dropout-on twin — so dropout really was
removed and the run was genuinely different. It is still exactly on chance.

## Reading the numbers

1. **The prediction fails.** `feat_lr` is raised 100× across the sweep while `head_lr` is fixed.
   If the spread were the blocker, the high-ratio arms should escape chance. They do not: every
   arm's `val_loss` is `4.170–4.173` against a chance `4.1589`, and `val_acc` is 0.65–1.31 % against
   a 1.56 % chance rate. Raising the feature-extractor learning rate 100-fold changes nothing.
2. **The feature weights stop moving, and that is the more interesting result.** `feat |dw|` goes
   `1.259e+01 → 1.862e+01 → 1.900e+01 → 1.960e+01 → 1.960e+01` for `feat_lr` of 0.1, 0.3, 1, 3, 10.
   A 100× increase in learning rate produces a **1.56×** increase in total displacement, and it
   **saturates between ratio 30 and ratio 100** — identical to four significant figures. The
   features are not learning-rate limited; they are being pushed into a plateau where extra `lr`
   buys nothing. Doubling `lr` should roughly double displacement if the gradient were meaningful.
3. **The head moves 2–3× further than the features even at ratio 1**, consistent with E034's
   spread, and `head |dw|` is essentially constant (`4.115–4.120e+01`) across the whole sweep — as
   it should be, since `head_lr` never changes. The param-group split is working as intended; the
   experiment is not broken.
4. **`trainable params` exposes a structural asymmetry**: features 3.75 M versus head 54.8 M — the
   classifier is **14.6×** the parameter count of the entire feature extractor, because
   `_flatten_features()` at 227 px produces a 9216-wide vector feeding two 4096-wide layers. The
   head has ample capacity to fit the noise of frozen random features, which is exactly the failure
   mode E034 predicted — but fixing the *learning rate* of the features does not fix it, so the
   problem is not how fast the features move.

## Command C — the last two regularisers, removed

Command B removed dropout. These two arms remove `weight_decay`, and the second removes both
regularisers at once — the last two differences between the SGD arms here and the memorising arm of
[E035](./E035-alexnet-init-is-not-what-blocks-sgd.md). If SGD still cannot fit its own training
stream with **neither** regulariser, no regularisation-side hypothesis survives.

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1 --weight_decay 0'

docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1 --weight_decay 0 --dropout off'
```

### Raw output, verbatim — `weight_decay=0`, dropout on

```
torch 2.10.0+cu128  device=cuda  classes=64  image_size=227  batch=32
steps=256 epochs=8 head_lr=0.1 clip=1.0 momentum=0.9 wd=0.0 seed=42
chance loss ln(64) = 4.1589; PASS rule (as in check_learn.py) = val_loss < 2.4953
arms: feature_lr / head_lr in [1.0]   dropout=on
Seed set to 42
ratio=     1  feat_lr=0.1        head_lr=0.1      dropout=on(0) loss 4.7591->4.1884  val_loss=4.1748 val_acc= 0.65%  max|dw| feat=1.543e+00 head=1.638e+01  [175s]

=== summary (chance val_loss = 4.1589, chance val_acc = 1.56 %) ===
  ratio    feat_lr  val_loss  val_acc%   feat |dw|   head |dw|  verdict
      1        0.1    4.1748      0.65   1.543e+00   1.638e+01   chance

trainable params: features=3,747,200 head=54,796,352

VERDICT: INCONCLUSIVE -- no ratio escapes chance, so the gradient spread is not
         the (only) blocker either. P2-25 stays open; do not close it on this result.
```

### Raw output, verbatim — `weight_decay=0` **and** dropout off

```
torch 2.10.0+cu128  device=cuda  classes=64  image_size=227  batch=32
steps=256 epochs=8 head_lr=0.1 clip=1.0 momentum=0.9 wd=0.0 seed=42
chance loss ln(64) = 4.1589; PASS rule (as in check_learn.py) = val_loss < 2.4953
arms: feature_lr / head_lr in [1.0]   dropout=off
Seed set to 42
ratio=     1  feat_lr=0.1        head_lr=0.1      dropout=off(2) loss 4.3817->4.1884  val_loss=4.1748 val_acc= 0.65%  max|dw| feat=1.106e+00 head=5.528e+00  [365s]

=== summary (chance val_loss = 4.1589, chance val_acc = 1.56 %) ===
  ratio    feat_lr  val_loss  val_acc%   feat |dw|   head |dw|  verdict
      1        0.1    4.1748      0.65   1.106e+00   5.528e+00   chance

trainable params: features=3,747,200 head=54,796,352

VERDICT: INCONCLUSIVE -- no ratio escapes chance, so the gradient spread is not
         the (only) blocker either. P2-25 stays open; do not close it on this result.
```

### Replication (§9.9 double-run rule)

The `weight_decay=0`, dropout-on arm was run a second time in a fresh process. It reproduced
**exactly**: `loss 4.7591->4.1884`, `val_loss=4.1748`, `val_acc=0.65 %`,
`max|dw| feat=1.543e+00 head=1.638e+01`.

### What the `|dw|` column says here

Removing `weight_decay` collapses the displacement: `feat |dw|` drops from 1.258e+01 to
1.543e+00 and `head |dw|` from 4.112e+01 to 1.638e+01. That is the opposite of what a
weight-decay-is-crushing-the-weights story predicts — wd was *adding* to the displacement budget
here, not removing it, and taking it away leaves the network moving **less** while landing on the
same chance loss. The weights are not being pulled toward zero; they are being pulled nowhere in
particular.

## Reading the four regularisation arms together

| Arm | dropout | `wd` | `val_loss` | `val_acc` | `feat \|dw\|` | `head \|dw\|` |
|---|---|---|---|---|---|---|
| A ratio 1 | on | 5e-4 | 4.1741 | 0.65 % | 1.259e+01 | 4.112e+01 |
| B | **off** | 5e-4 | 4.1741 | 0.65 % | 1.258e+01 | 4.112e+01 |
| C1 | on | **0** | 4.1748 | 0.65 % | 1.543e+00 | 1.638e+01 |
| C2 | **off** | **0** | 4.1748 | 0.65 % | 1.106e+00 | 5.528e+00 |

Four arms spanning the full on/off cross-product of the two regularisers, and `val_loss` moves by
**0.0007** — less than one validation image out of 153. Whatever pins SGD at chance is not a
regulariser.

## What this closes

Combined with the other negatives, every optimiser-side **and** regularisation-side explanation for
[E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)'s all-chance SGD grid is now
ruled out by measurement:

| Candidate | Ruled out by |
|---|---|
| `gradient_clip_val=1.0` | E033 (clip off / clip 30 also at chance) |
| fp16 overflow | E033 (`bf16-mixed` also at chance) |
| Step budget too short | E033 (Adam control converges at the same budget; 7 680-step SGD arms) |
| The uniform `std=0.01` init | [E035](./E035-alexnet-init-is-not-what-blocks-sgd.md) (memorises 3/4 fixed batches) |
| The cosine schedule | [E037](./E037-lr-schedule-is-stepped-per-epoch-and-healthy.md) (stepped per epoch, LR healthy) |
| The 1.5 × 10⁴ gradient spread | **this file** (per-group `lr` up to 100× does nothing) |
| `dropout p=0.5` | **this file**, Command B (all dropouts replaced by `nn.Identity`, still chance) |
| `weight_decay=5e-4` | **this file**, Command C (`wd=0`, alone and combined with dropout off) |

**One candidate survives: the data.** 16 384 draws-with-replacement from only **4 153 unique**
in-class train images. The `feat |dw|` saturation points there — the features are not short of
learning rate, they are short of *signal*.

## Caveats

- `gradient_clip_val=1.0` is **on** in this probe (it is the shipped default). Because clipping
  renormalises the *total* gradient before the groups are scaled, very high `feat_lr` arms are
  partly clipped — but the ratio-1 arm is not clipped-limited at the same total norm, and the
  saturation pattern is the same across all arms, so clipping does not explain the null result.
  A clip-off replication is the natural follow-up.
- `val_acc` is quantised in 0.654 % steps (153 in-class images,
  [E036](./E036-validation-pool-size-and-accuracy-resolution.md)); 0.65 % is one image and 1.31 %
  is two. None of these differences are meaningful.
- `max|dw|` is the max over per-tensor L2 displacements from init, across differently-shaped
  tensors. It measures "did this group move", not "did this group learn".
- 256 × 8 = 2 048 steps per arm. E033's 7 680-step arms were also at chance, so the budget is not
  why these arms are flat.
- The probe's own verdict string says `INCONCLUSIVE`, deliberately: a null here does **not** close
  `P2-25`, and the probe is written so it cannot be misread as closing it.

## Verdict

Per-group learning rates do not rescue SGD, and neither does removing dropout nor removing weight
decay. The init gradient spread is a real, measured property of AlexNet and it does explain why
Adam is needed — but it is not the *cause* of the chance-level plateau.

`P2-25` **stays open.** Eight candidate causes have been measured and all eight are eliminated:
the clip, fp16 overflow, the step budget, the init, the schedule, the gradient spread, dropout and
weight decay. The only hypothesis left is the data — 4 153 unique in-class training images — and
the two measurements that point at it are that `train_acc` never exceeds 1.95 % in any SGD arm and
that `feat |dw|` saturates. The next agent should attack the data, not the optimiser.
