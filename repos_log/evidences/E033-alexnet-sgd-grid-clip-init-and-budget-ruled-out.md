# E033 — AlexNet SGD: clipping, the `std=0.01` init and the step budget are all ruled out

| | |
|---|---|
| **Status** | CONFIRMED (negative result) — 9 SGD arms across clip × lr × precision, plus two 7 680-step arms (11 in total), all sit exactly on chance; Adam at the *identical* budget leaves chance. **`P2-25` is NOT closed.** |
| **Cards** | `P2-25` (re-tested, stays open), supersedes the *causal claim* in [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md) |
| **Command** | see §Commands — every block is copy-pasteable |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists

[E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md) closed its SGD arm with a causal
claim: `gradient_clip_val=1.0` scales every step by 0.037 against an init total gradient norm of
26.9, therefore the clip is why SGD sits at chance, and *"a fair SGD run needs the clip raised
above the init norm first"*. `P2-25` was opened to test exactly that. The prediction is precise:
raise the clip above 26.9 (or switch it off) and SGD should leave chance.

This file tests it. It does not, because the prediction fails — and the failure is not explained
by the init, nor by the number of steps either.

## Setup

All arms share everything except the named knob: 64 classes, 227 px, `batch_size=32`,
`train_samples_per_epoch=16384`, `momentum=0.9`, `weight_decay=5e-4`, `scheduler=cosine`,
`seed=42`, `16-mixed` unless stated, on the canonical
`ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538` cache. Chance for 64-class cross-entropy is
`ln(64) = 4.1589`; the chance accuracy rate is `1/64 = 1.56 %`.

## Command A — the 9-arm clip × lr × precision grid

```bash
docker exec ppg-dev6 bash -lc '
  mkdir -p /tmp/ppglogs /tmp/ck_p225
  run() {
    local tag="$1" clip="$2" lr="$3" prec="${4:-16-mixed}"
    rm -rf "/tmp/ck_p225/$tag"
    python AlexNet/train.py \
      --version_name "p225_$tag" \
      --num_classes 64 --image_size 227 \
      --optimizer sgd --scheduler cosine \
      --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
      --batch_size 32 --eval_batch_size 64 \
      --max_epochs 5 --lr "$lr" --momentum 0.9 --weight_decay 5e-4 \
      --precision "$prec" --accelerator gpu --num_workers 8 --seed 42 \
      --gradient_clip_val "$clip" --limit_train_batches 200 \
      --checkpoint_path "/tmp/ck_p225/$tag" --log_path /tmp/lg_p225 \
      > "/tmp/ppglogs/p225_$tag.log" 2>&1
    tr -d "\r" < "/tmp/ppglogs/p225_$tag.log" \
      | grep -aoE "val_loss=[0-9.nan]+, val_acc=[0-9.]+, train_loss_epoch=[0-9.nan]+, train_acc=[0-9.]+" \
      | awk "!seen[\$0]++"
  }
  run clip1_lr0p1        1.0  0.1
  run noclip_lr0p1       0    0.1
  run noclip_lr3e-2      0    3e-2
  run noclip_lr1e-2      0    1e-2
  run noclip_lr3e-3      0    3e-3
  run noclip_lr1e-3      0    1e-3
  run clip30_lr0p1       30   0.1
  run clip30_lr3e-2      30   3e-2
  run noclip_bf16_lr0p1  0    0.1   bf16-mixed'
```

### Raw output — final epoch of every arm, verbatim

```
clip1_lr0p1                val_loss=4.160, val_acc=2.610, train_loss_epoch=4.160, train_acc=1.530
noclip_lr0p1               val_loss=nan.0, val_acc=1.310, train_loss_epoch=nan.0, train_acc=1.950
noclip_lr3e-2              val_loss=4.160, val_acc=2.610, train_loss_epoch=4.160, train_acc=1.590
noclip_lr1e-2              val_loss=4.160, val_acc=1.310, train_loss_epoch=4.160, train_acc=1.410
noclip_lr3e-3              val_loss=4.160, val_acc=2.610, train_loss_epoch=4.160, train_acc=1.380
noclip_lr1e-3              val_loss=4.160, val_acc=2.610, train_loss_epoch=4.180, train_acc=1.610
clip30_lr0p1               val_loss=4.160, val_acc=2.610, train_loss_epoch=4.160, train_acc=1.720
clip30_lr3e-2              val_loss=4.160, val_acc=2.610, train_loss_epoch=4.160, train_acc=1.270
noclip_bf16_lr0p1          val_loss=4.150, val_acc=2.610, train_loss_epoch=4.160, train_acc=1.730
```

`clip1_lr0p1` reproduces E031's SGD arm. `noclip_lr0p1` is E031's arm with the clip removed: it
**NaNs by step ~5** (`4.52 → 54.2 → 51.2 → nan`), which is E031's own prediction confirmed — the
clip *is* what keeps `lr=0.1` finite. But the clip is also what keeps it *finite and useless*:
every arm that stays finite has `val_loss ≈ 4.16 = ln(64)` and `train_acc` in 1.27–1.73 %, i.e.
inside the 1.56 % chance band. Removing the clip does not rescue SGD at any `lr` from `1e-3` to
`3e-2`; it only changes *where* the arm dies. `bf16-mixed` changes nothing either, so this is not
fp16 overflow (the §4.4 "try bf16 before suspecting the model" check).

## Command B — the Adam control at the identical budget

An all-chance grid is only evidence about SGD if the *budget* is sufficient for some optimiser.
Otherwise "200 steps × 5 epochs is simply too short" explains everything.

```bash
docker exec ppg-dev6 bash -lc '
  rm -rf /tmp/ck_p225/ctrl_adam200x5 &&
  python AlexNet/train.py \
    --version_name p225_ctrl_adam200x5 \
    --num_classes 64 --image_size 227 \
    --optimizer adam --scheduler cosine \
    --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
    --batch_size 32 --eval_batch_size 64 \
    --max_epochs 5 --lr 3e-4 --momentum 0.9 --weight_decay 5e-4 \
    --precision 16-mixed --accelerator gpu --num_workers 8 --seed 42 \
    --gradient_clip_val 1.0 --limit_train_batches 200 \
    --checkpoint_path /tmp/ck_p225/ctrl_adam200x5 --log_path /tmp/lg_p225 \
    > /tmp/ppglogs/p225_ctrl_adam200x5.log 2>&1
  tr -d "\r" < /tmp/ppglogs/p225_ctrl_adam200x5.log \
    | grep -aoE "val_loss=[0-9.nan]+, val_acc=[0-9.]+, train_loss_epoch=[0-9.nan]+, train_acc=[0-9.]+" \
    | awk "!seen[\$0]++"'
```

### Raw output — Adam control, verbatim (each epoch appears twice: before and after the val pass)

```
val_loss=4.120, val_acc=0.654, train_loss_epoch=4.470, train_acc=1.700
val_loss=4.040, val_acc=1.310, train_loss_epoch=4.470, train_acc=1.700
val_loss=4.040, val_acc=1.310, train_loss_epoch=4.120, train_acc=2.890
val_loss=3.990, val_acc=5.230, train_loss_epoch=4.120, train_acc=2.890
val_loss=3.990, val_acc=5.230, train_loss_epoch=4.050, train_acc=4.110
val_loss=3.900, val_acc=5.230, train_loss_epoch=4.050, train_acc=4.110
val_loss=3.900, val_acc=5.230, train_loss_epoch=3.960, train_acc=5.330
val_loss=3.850, val_acc=7.190, train_loss_epoch=3.960, train_acc=5.330
val_loss=3.850, val_acc=7.190, train_loss_epoch=3.930, train_acc=6.310
```

**The budget is sufficient.** Adam moves `val_loss` 4.120 → 3.850 and `val_acc` 0.654 → 7.190 %
(4.6× chance) in exactly the steps where all nine SGD arms did not move at all. Re-run a second
time in a fresh process, it reproduced to the digit. So the grid's all-chance result is a finding
about SGD, not an artefact of a short run.

## Command C — the two most promising arms, at fair length

```bash
docker exec ppg-dev6 bash -lc '
  for spec in "sgd_clip30_long 30 0.1" "sgd_noclip_long 0 1e-2"; do
    set -- $spec; tag=$1; clip=$2; lr=$3
    rm -rf "/tmp/ck_p225/$tag"
    python AlexNet/train.py \
      --version_name "p225_$tag" \
      --num_classes 64 --image_size 227 \
      --optimizer sgd --scheduler cosine \
      --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
      --batch_size 32 --eval_batch_size 64 \
      --max_epochs 15 --lr "$lr" --momentum 0.9 --weight_decay 5e-4 \
      --precision 16-mixed --accelerator gpu --num_workers 8 --seed 42 \
      --gradient_clip_val "$clip" --limit_train_batches 512 \
      --checkpoint_path "/tmp/ck_p225/$tag" --log_path /tmp/lg_p225 \
      > "/tmp/ppglogs/p225_$tag.log" 2>&1
    tr -d "\r" < "/tmp/ppglogs/p225_$tag.log" \
      | grep -aoE "val_loss=[0-9.nan]+, val_acc=[0-9.]+, train_loss_epoch=[0-9.nan]+, train_acc=[0-9.]+" \
      | awk "!seen[\$0]++"
  done'
```

### Raw output — last three epochs of `sgd_clip30_long` (512 × 15 = 7 680 steps), verbatim

```
val_loss=4.160, val_acc=1.310, train_loss_epoch=4.160, train_acc=1.720
val_loss=4.160, val_acc=1.310, train_loss_epoch=4.160, train_acc=1.850
val_loss=4.160, val_acc=1.310, train_loss_epoch=4.160, train_acc=1.770
```

`sgd_noclip_long` (clip off, `lr=1e-2`, 7 680 steps) ends identically: `val_loss=4.150`,
`val_acc=2.610`, `train_acc=1.880`. 7 680 steps is 7.7× the grid and comparable to E031's
*converged* Adam arm (512 × 12). SGD is still on chance.

## What this rules out, and what it does not

| Hypothesis | Test | Result |
|---|---|---|
| The clip blocks SGD (E031's claim) | clip ∈ {1.0, 30, off} | **Refuted.** Off and 30 are also exactly at chance. |
| fp16 overflow | `bf16-mixed` arm | **Refuted.** Same chance. |
| Too few steps | Adam control at same budget; 7 680-step SGD arms | **Refuted.** Adam converges at the budget; SGD does not at 7.7× it. |
| The uniform `std=0.01` init | [E035](./E035-alexnet-init-is-not-what-blocks-sgd.md) | **Refuted.** The paper init *memorises* a fixed batch with unclipped SGD. |
| The cosine schedule | [E037](./E037-lr-schedule-is-stepped-per-epoch-and-healthy.md) | **Refuted.** Measured LR trajectory is healthy. |
| The 1.5e4 per-tensor gradient spread | [E038](./E038-per-group-lr-sgd-sweep.md) | **Refuted.** Per-group `lr` up to 100x does nothing. |
| `dropout p=0.5` | [E038](./E038-per-group-lr-sgd-sweep.md), Command B | **Refuted.** Every `nn.Dropout` replaced by `nn.Identity`, still chance. |
| `weight_decay=5e-4` | [E038](./E038-per-group-lr-sgd-sweep.md), Command C | **Refuted.** `wd=0` at chance, alone and combined with dropout off. |

**What is left.** SGD *can* fit these images (E035: loss 2.3025 → 0.0003 on a fixed batch) but
does not *generalise* them, while Adam does. The remaining candidates are the regularisers and the
data: `dropout p=0.5` plus `weight_decay=5e-4` acting on 16 384 draws-with-replacement from only
**4 153 unique** in-class train images, and augmentation strength. Note that `train_acc` never
exceeds 1.95 % in any SGD arm — the network is not even fitting its *training* stream, which is
what makes the "SGD can fit but not generalise" framing incomplete and keeps `P2-25` open.

## Caveats

- `val_acc` here is a multiple of `100/153 = 0.6536 %` (0.654, 1.310, 1.960, 2.610, 5.230, 7.190 —
  all visible above). Differences of one or two images are not signal; see
  [E036](./E036-validation-pool-size-and-accuracy-resolution.md) and `P2-27`.
- A NaN arm still exits 0 and still writes `alexnet_final.ckpt`: `EarlyStopping` has a finite
  guard and `divergence_threshold=50.0`, so it ends the run at epoch 0. **Grep the log; do not
  trust the exit code.**
- Lightning's progress bar writes `\r`; the `tr -d "\r" | grep -aoE ... | awk "!seen[\$0]++"`
  pipeline above is what makes these numbers extractable. Each epoch appears twice.
- These arms used `--val_samples 2048` because that is the cache E031 used, so this file is
  directly comparable with E031. It is *not* comparable with runs on the `P2-27` 16 384 pool,
  which selects 8 different classes.

## Verdict

`gradient_clip_val=1.0` is **not** what pins AlexNet's SGD arm at chance. E031's causal claim is
superseded; its measurements (init grad norm 26.9, Adam converging, SGD at chance) all stand and
were reproduced. `P2-25`'s acceptance criterion is *"an SGD run that beats chance"* — no arm here
does, so the card **stays open** and Adam remains the shipped optimiser.
