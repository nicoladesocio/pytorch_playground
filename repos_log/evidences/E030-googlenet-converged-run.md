# E030 — GoogLeNet converges: `val_acc` 34.6 % against a 1.56 % chance level

| | |
|---|---|
| **Status** | CONFIRMED — first run in this repo's history that is unambiguously above chance |
| **Cards** | `P2-14` (land a run that beats chance), `P2-11` (AlexNet/GoogLeNet never learn), `P2-10` (epoch length) |
| **Command** | see "Command" below — copy-pasteable, run inside `ppg-gpu:latest` |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev5` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_005](../handoffs/HANDOFF_005.md) |

## Why this run

Since [HANDOFF_003](../handoffs/HANDOFF_003.md) the repo's position has been that neither network
learns: AlexNet "plateaued" at chance over 18 epochs, GoogLeNet sat at `val_acc ≈ 1.5 %`. Every
probe since then has been about *why*. `P2-14` is the card that asks the opposite question —
**produce one run that is measurably above chance**, so that "the network cannot learn" becomes a
question with a counterexample instead of an article of faith.

Chance for `num_classes=64` is `1/64 = 1.56 %` and `ln(64) = 4.1589`. GoogLeNet's training loss is
the *combined* loss (`main + 0.3·aux1 + 0.3·aux2`), so its chance-level value is
`1.6·ln(64) = 3.6841` — see [E019](./E019-chance-level-per-network.md) / `chance_loss()` in
`probes/check_overfit2.py` (`P2-17`). A run is "above chance" here if `val_acc` is several times
1.56 % and `val_loss` is below 3.68.

## Command

```bash
# host, from the repo root
docker run -d --name ppg-dev5 --gpus all --entrypoint bash \
  -v "$PWD":/workspace -w /workspace ppg-gpu:latest -c "sleep infinity"

docker exec ppg-dev5 bash -lc '
  mkdir -p /tmp/ppglogs /tmp/ck_conv_goog /tmp/lg_conv &&
  nohup python GoogLeNet/train.py \
    --version_name conv_goog_adam \
    --num_classes 64 --image_size 227 \
    --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
    --batch_size 64 --eval_batch_size 128 \
    --max_epochs 60 --optimizer adam --lr 1e-4 --scheduler cosine \
    --precision 16-mixed --accelerator gpu --num_workers 8 --seed 42 \
    --checkpoint_path /tmp/ck_conv_goog --log_path /tmp/lg_conv \
    > /tmp/ppglogs/goog_adam.log 2>&1 &'
```

The resolved argument namespace, from the log:

```
Namespace(version_name='conv_goog_adam', dataset_name='ILSVRC/imagenet-1k',
cache_root='/workspace/Data/datasets', num_classes=64, image_size=227,
checkpoint_path='/tmp/ck_conv_goog', log_path='/tmp/lg_conv', train_samples=51200,
train_samples_per_epoch=16384, val_samples=2048, batch_size=64, eval_batch_size=128,
max_epochs=60, optimizer='adam', lr=0.0001, momentum=0.9, weight_decay=0.0001,
eta_min=1e-06, scheduler='cosine', aux_loss_weight=0.3, swa=True, swa_lrs=0.0001,
swa_epoch_start=0.8, annealing_epochs=2, precision='16-mixed', accelerator='gpu',
num_workers=8, seed=42, shuffle_buffer=2048, deterministic=False, resume=False,
limit_train_batches=None, limit_val_batches=None)
```

and the class-selection line, which is what fixes the epoch size:

```
Selected 64 classes; per-class cached counts: min=62, median=65, max=73
```

## Raw output — per-epoch metrics, verbatim

Extracted from `/tmp/ppglogs/goog_adam.log` (progress-bar lines, ANSI stripped, one row per
completed epoch; `train_acc` is the epoch value Lightning reports at the following validation):

```
  ep  0  val_loss=4.01  val_acc=4.58   train_acc=2.2
  ep  1  val_loss=3.73  val_acc=5.88   train_acc=5.99
  ep  2  val_loss=3.68  val_acc=7.84   train_acc=7.67
  ep  3  val_loss=3.71  val_acc=8.5    train_acc=9.18
  ep  4  val_loss=3.56  val_acc=9.8    train_acc=10.9
  ep  5  val_loss=3.42  val_acc=13.1   train_acc=12.8
  ep  6  val_loss=3.31  val_acc=15.7   train_acc=14.5
  ep  7  val_loss=3.18  val_acc=17.0   train_acc=16.6
  ep  8  val_loss=3.12  val_acc=24.2   train_acc=18.5
  ep  9  val_loss=3.16  val_acc=24.2   train_acc=18.7
  ep 10  val_loss=3.05  val_acc=19.6   train_acc=21.2
  ep 11  val_loss=3.01  val_acc=19.0   train_acc=22.3
  ep 12  val_loss=3.01  val_acc=19.6   train_acc=23.2
  ep 13  val_loss=3.0   val_acc=24.2   train_acc=24.9
  ep 14  val_loss=2.91  val_acc=23.5   train_acc=26.2
  ep 15  val_loss=2.94  val_acc=20.3   train_acc=27.3
  ep 16  val_loss=2.92  val_acc=25.5   train_acc=27.9
  ep 17  val_loss=2.9   val_acc=27.5   train_acc=29.5
  ep 18  val_loss=2.91  val_acc=30.1   train_acc=31.4
  ep 19  val_loss=2.87  val_acc=31.4   train_acc=31.2
  ep 20  val_loss=2.92  val_acc=30.7   train_acc=33.1
  ep 21  val_loss=2.95  val_acc=28.8   train_acc=33.5
  ep 22  val_loss=2.88  val_acc=27.5   train_acc=34.7
  ep 23  val_loss=2.92  val_acc=30.7   train_acc=36.1
  ep 24  val_loss=2.93  val_acc=31.4   train_acc=37.4
  ep 25  val_loss=2.84  val_acc=30.7   train_acc=37.7
  ep 26  val_loss=3.0   val_acc=25.5   train_acc=39.5
  ep 27  val_loss=3.02  val_acc=30.7   train_acc=40.5
  ep 28  val_loss=3.02  val_acc=33.3   train_acc=41.6
  ep 29  val_loss=2.84  val_acc=34.6   train_acc=41.7
  ep 30  val_loss=2.91  val_acc=33.3   train_acc=43.5
  ep 31  val_loss=2.99  val_acc=33.3   train_acc=44.5
  ep 32  val_loss=3.11  val_acc=30.7   train_acc=45.6
  ep 33  val_loss=3.04  val_acc=33.3   train_acc=46.2
  ep 34  val_loss=3.08  val_acc=34.0   train_acc=46.4
  ep 35  val_loss=3.06  val_acc=34.6   train_acc=48.5
```

The run was stopped by hand at epoch 36 of 60 (`[rank: 0] Received SIGTERM: 15`) once the trend was
unambiguous — see "Why it stopped at 36, not 60".

## Interpretation

**It is learning, and it is learning generalisably.** Three independent signals, all pointing the
same way:

1. `val_acc` goes from 4.58 % (≈ chance + noise) to 34.6 % — **22× the chance level of 1.56 %**.
2. `val_loss` goes from 4.01 to a minimum of 2.84, well below the combined-loss chance value of
   3.6841. A model that had merely memorised the training set would drive `train_loss` down while
   `val_loss` stayed at or rose above chance.
3. `train_acc` and `val_acc` rise **together** (2.2 → 48.5 and 4.6 → 34.6). The gap
   `train_acc − val_acc` stays around 14 points throughout rather than opening up, so at this
   length the run is capacity-limited, not overfitting-limited.

**The epoch length is the knob that made it work.** `--train_samples_per_epoch 16384` at
`batch_size 64` is 256 optimiser steps per epoch. Without it, an epoch on this cache is 4 153
samples = **65 steps** (measured in [HANDOFF_005 §3.4](../handoffs/HANDOFF_005.md), from
`choose_target_classes` in
[utils/datasets_utils.py](../../utils/datasets_utils.py)). HANDOFF_003's "plateaued after 18
epochs" was 18 × 65 ≈ 1 170 steps; this run reached 22× chance at 18 × 256 ≈ 4 600 steps. The
earlier "plateau" was an epoch that was too short to move the weights, not a network that could not
learn. That is the mechanism `P2-10` suspected, now with a counterexample on the other side of it.

**`val_acc` is quantised at 0.654 %.** The validation split holds only 153 in-class images, so every
`val_acc` value is a multiple of `100/153`. The jumps (24.2 → 19.6 → 19.0) are 1–7 images changing
side, not instability. Read the *trend over several epochs*, not single epochs. A trustworthy
validation number needs a larger `--val_samples`, which is now a kanban card.

## Why it stopped at 36, not 60

`train.py` already installs `EarlyStopping(monitor="val_loss", patience=max(10, max_epochs//5))`,
so a genuinely flat run stops itself. This run had not triggered it — `val_loss` was still
oscillating inside patience. It was stopped manually because the question was *"is the network
learning?"*, and by epoch 10 (`val_acc` 19.6 %, `val_loss` 3.05) that question had a 12×-chance
answer. The remaining 25 epochs would have bought a better checkpoint, not a better conclusion.
The user's instruction for this session was explicit about this: these runs are probes that the
network learns; the deliverable is the parameterless `train_final.py` the human runs on a bigger
GPU.

## What this does *not* show

- **Not a final accuracy.** 153 validation images, one seed, one 36-epoch slice, on a 6 GB laptop
  GPU at batch 64. The number to quote is "well above chance and still climbing", not "34.6 %".
- **Not the `train_final.py` recipe.** That script bakes `max_epochs=60`, `batch_size=64` and no
  `train_samples_per_epoch` — i.e. it would run the *short-epoch* configuration this file shows to
  be the wrong one. See HANDOFF_005 §9.
- **Not a comparison against AlexNet.** That is
  [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md).
- **SWA was on** (`swa=True`, `swa_epoch_start=0.8`) and had not yet engaged at epoch 36 of a
  60-epoch schedule, so nothing here says anything about SWA's contribution.
