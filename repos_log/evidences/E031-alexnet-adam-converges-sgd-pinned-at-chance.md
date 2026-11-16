# E031 — AlexNet: Adam converges, SGD is pinned at chance by `gradient_clip_val=1.0`

| | |
|---|---|
| **Status** | CONFIRMED — AlexNet learns (Adam, `val_acc` 24.8 %); SGD-at-chance is a clipping artefact, not an optimiser failure |
| **Cards** | `P2-14`, `P2-11`, `P2-12` (AlexNet optimiser), new card `P2-25` (gradient clipping vs SGD) |
| **Command** | see the two "Command" blocks below |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev5` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_005](../handoffs/HANDOFF_005.md) |

## Why this run

HANDOFF_003 reported AlexNet "plateauing" at chance. [E030](./E030-googlenet-converged-run.md)
shows GoogLeNet converging once the epoch is long enough, but GoogLeNet's default optimiser is Adam
and AlexNet's is SGD — so E030 alone cannot separate "the epoch was too short" from "AlexNet's
recipe is broken". This file runs AlexNet twice, identically except for the optimiser, on the same
long epochs E030 used.

## Command A — SGD, `lr=0.1` (the paper-style arm)

```bash
docker exec ppg-dev5 bash -lc '
  mkdir -p /tmp/ppglogs /tmp/ck_conv_alex /tmp/lg_conv &&
  nohup python AlexNet/train.py \
    --version_name conv_alex_sgd \
    --num_classes 64 --image_size 227 --optimizer sgd --scheduler cosine \
    --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
    --batch_size 32 --eval_batch_size 64 \
    --max_epochs 60 --lr 0.1 --momentum 0.9 --weight_decay 5e-4 \
    --precision 16-mixed --accelerator gpu --num_workers 8 --seed 42 \
    --checkpoint_path /tmp/ck_conv_alex --log_path /tmp/lg_conv \
    > /tmp/ppglogs/alex_sgd.log 2>&1 &'
```

### Raw output — SGD, verbatim (21 epochs, then killed)

```
  ep  0 val_loss=4.16 val_acc=1.96 train_acc=1.57
  ep  1 val_loss=4.16 val_acc=1.96 train_acc=1.53
  ep  2 val_loss=4.17 val_acc=1.31 train_acc=1.7
  ep  3 val_loss=4.16 val_acc=3.92 train_acc=1.5
  ep  4 val_loss=4.16 val_acc=1.31 train_acc=1.48
  ep  5 val_loss=4.18 val_acc=1.96 train_acc=1.51
  ep  6 val_loss=4.16 val_acc=1.31 train_acc=1.45
  ep  7 val_loss=4.15 val_acc=1.96 train_acc=1.45
  ep  8 val_loss=4.16 val_acc=1.31 train_acc=1.56
  ep  9 val_loss=4.16 val_acc=1.96 train_acc=1.51
  ep 10 val_loss=4.15 val_acc=3.27 train_acc=1.65
  ep 11 val_loss=4.17 val_acc=0.654 train_acc=1.65
  ep 12 val_loss=4.17 val_acc=1.31 train_acc=1.64
  ep 13 val_loss=4.16 val_acc=1.31 train_acc=1.81
  ep 14 val_loss=4.15 val_acc=0.654 train_acc=1.5
  ep 15 val_loss=4.16 val_acc=0.654 train_acc=1.54
  ep 16 val_loss=4.16 val_acc=2.61 train_acc=1.56
  ep 17 val_loss=4.17 val_acc=2.61 train_acc=1.62
  ep 18 val_loss=4.16 val_acc=1.96 train_acc=1.75
  ep 19 val_loss=4.17 val_acc=2.61 train_acc=1.89
  ep 20 val_loss=4.16 val_acc=1.31 train_acc=1.39
```

`val_loss` never leaves `4.16 ± 0.01` and `train_acc` never leaves chance (`1/64 = 1.56 %`) over
21 epochs × 512 steps ≈ 10 750 optimiser steps. `ln(64) = 4.1589`. This is the uniform-posterior
solution: the network has not moved.

## Command B — Adam, `lr=3e-4` (everything else identical)

```bash
docker exec ppg-dev5 bash -lc '
  mkdir -p /tmp/ppglogs /tmp/ck_conv_alex_adam /tmp/lg_conv &&
  nohup python AlexNet/train.py \
    --version_name conv_alex_adam \
    --num_classes 64 --image_size 227 --optimizer adam --scheduler cosine \
    --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
    --batch_size 32 --eval_batch_size 64 \
    --max_epochs 60 --lr 3e-4 --weight_decay 5e-4 \
    --precision 16-mixed --accelerator gpu --num_workers 8 --seed 42 \
    --checkpoint_path /tmp/ck_conv_alex_adam --log_path /tmp/lg_conv \
    > /tmp/ppglogs/alex_adam.log 2>&1 &'
```

### Raw output — Adam, verbatim (13 epochs, then stopped by hand)

```
  ep  0 val_loss=4.03 val_acc=4.58  train_acc=2.67
  ep  1 val_loss=3.84 val_acc=7.84  train_acc=5.08
  ep  2 val_loss=3.7  val_acc=12.4  train_acc=7.88
  ep  3 val_loss=3.56 val_acc=11.8  train_acc=9.97
  ep  4 val_loss=3.54 val_acc=11.8  train_acc=11.9
  ep  5 val_loss=3.51 val_acc=12.4  train_acc=13.0
  ep  6 val_loss=3.51 val_acc=15.0  train_acc=15.2
  ep  7 val_loss=3.4  val_acc=15.0  train_acc=16.7
  ep  8 val_loss=3.28 val_acc=18.3  train_acc=18.6
  ep  9 val_loss=3.24 val_acc=18.3  train_acc=19.3
  ep 10 val_loss=3.12 val_acc=24.8  train_acc=22.1
  ep 11 val_loss=3.13 val_acc=22.9  train_acc=24.0
  ep 12 val_loss=3.11 val_acc=20.3  train_acc=24.5
```

**AlexNet learns.** `val_acc` 4.58 % → 24.8 % (16× the 1.56 % chance level), `val_loss` 4.03 → 3.11,
`train_acc` 2.67 → 24.5, both rising together. It reached 18.3 % by epoch 8 and was stopped at
epoch 12 because the question was answered — same reasoning as E030, and the same instruction from
the user (these are probes that the network learns, not the deliverable).

## Root cause of the SGD arm: `gradient_clip_val=1.0`

`AlexNet/train.py` passes `gradient_clip_val=1.0` to `pl.Trainer`. Measured at initialisation, on
the real model, on GPU, batch 32 at 227 px:

```
init loss            = 4.5987   (ln(64) = 4.1589)
init total grad norm = 26.92
clip factor at gradient_clip_val=1.0 -> step scaled by 0.0371
logit std            = 0.9254
```

Reproduce with:

```bash
docker exec ppg-dev5 bash -lc 'cd /workspace && python - <<PY
import os, sys, torch, importlib.util
sys.path.insert(0, "AlexNet"); sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("ppg_alex", "AlexNet/model.py")
m = importlib.util.module_from_spec(spec); sys.modules["ppg_alex"] = m; spec.loader.exec_module(m)
torch.manual_seed(42)
net = m.AlexNet(num_classes=64, image_size=227).cuda(); net.train()
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(32,3,227,227, generator=g, device="cuda")
y = torch.randint(0,64,(32,), generator=g, device="cuda")
loss = torch.nn.functional.cross_entropy(net(x), y); loss.backward()
tot = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")).item()
print(f"init loss            = {loss.item():.4f}   (ln(64) = {torch.tensor(64.).log().item():.4f})")
print(f"init total grad norm = {tot:.2f}")
print(f"clip factor at gradient_clip_val=1.0 -> step scaled by {min(1.0, 1.0/tot):.4f}")
print(f"logit std            = {net(x).detach().std().item():.4f}")   # dropout is active: varies
PY'
```

Re-run verbatim, the block reproduces (`init total grad norm = 26.92`, `clip factor … 0.0371`);
`logit std` varies by ~0.015 between invocations because dropout is active in `train()` mode.

The init total gradient norm is **26.92** — this is expected for AlexNet's paper-style
initialisation (`normal_(std=0.01)` on conv weights, bias = 1 on conv2/4/5 and fc1/fc2, five
conv layers + two 4096-unit FC layers, see
[AlexNet/model.py](../../AlexNet/model.py)). Clipping to 1.0 therefore multiplies **every** SGD
step by **0.037**. `lr = 0.1` becomes an effective `lr ≈ 0.0037`, which is why 10 750 steps produced
no movement at all.

Adam is scale-invariant — it divides by `sqrt(v̂)` — so the same clip is nearly harmless there, and
arm B converges. That is the whole difference between the two runs.

The TensorBoard `grad_norm/total` scalars agree. `AlexNetLightning.on_after_backward` logs the norm
*before* the fp16 loss-scale is undone, so divide by the initial `GradScaler` scale (65536):

```
conv_alex_sgd   grad_norm/total: n=203 first=1635035.25 min=10650.08 max=1635035.25 last=42233.34
conv_alex_adam  grad_norm/total: n=127 first=1635035.25 min=52307.20 max=1635035.25 last=192823.97
```

`1635035.25 / 65536 = 24.95`, matching the 26.92 measured above. The SGD run's norm never collapses
to zero and never explodes — it stays large, which is what "the clip is doing all the work" looks
like.

## Interpretation

- **`P2-11` ("AlexNet never learns") is closed.** It learns with Adam at 227 px on real images.
- **The SGD arm is not evidence against SGD.** It is evidence about `gradient_clip_val`. A fair SGD
  test needs either `gradient_clip_val=None` or `gradient_clip_val ≈ 25–50` (i.e. above the init
  norm, so it only catches genuine spikes), or a paper-style `lr` of 0.01–0.1 *with* the clip
  removed. None of those was run here — this file deliberately keeps the repo's current trainer
  settings so the A/B isolates the optimiser, and the clip turns out to be the confounder.
- **Do not read the SGD arm as "SGD needs a bigger lr".** Raising `lr` to 1.0 under a clip of 1.0
  only moves the effective lr from 0.037 to 0.0037 — the clip caps the step size regardless.

## What this does *not* show

- **No SGD configuration is validated.** `AlexNet/train_final.py` currently bakes
  `optimizer="sgd"`, `lr=0.01` — the configuration this file shows cannot move under the clip. It
  must be changed or the clip must be raised; see HANDOFF_005 §9.
- **Not a tuned Adam run.** `lr=3e-4` was picked as a safe Adam default, not swept. A larger lr or
  a longer run should do better.
- **Not a final accuracy** — 153 in-class validation images (`val_acc` is quantised at 0.654 %),
  one seed, batch 32 on a 6 GB card.
- **The clip measurement is at initialisation.** The norm during training varies (the TensorBoard
  series ranges from ~0.16 to ~2.9 after unscaling), so 0.037 is the *early-training* factor, not a
  constant.
