# E037 — AlexNet's cosine schedule is stepped per **epoch** and the LR stays healthy

| | |
|---|---|
| **Status** | CONFIRMED (negative result) — the schedule is **not** the reason SGD sits at chance |
| **Cards** | `P2-25` (rules out a candidate cause) |
| **Command** | `docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_lr_schedule.py --optimizer sgd --lr 0.01 --scheduler cosine --epochs 2 --steps 300'` |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists — and why it is written as a refutation of its author

After [E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md) ruled out the clip and
[E034](./E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md) ruled out the step
budget, the next candidate was the learning-rate *schedule*, and the source appeared to hand it to
me on a plate. `AlexNet/model.py::configure_optimizers` does:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon, eta_min=0.0)
return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

with `horizon = self.trainer.max_epochs`. Lightning's documented default for a *bare* scheduler
entry is `interval="step", frequency=1` — advanced once per **optimizer step**. So `T_max` looked
like it was being *interpreted* in steps while being *supplied* in epochs: with `max_epochs=60` the
cosine would complete its entire decay in the first 60 batches of a ~512-batch epoch, leaving
`eta_min = 0.0` for the remaining 99 % of every epoch. That story is coherent, explains a lot, and
**is wrong**.

This file exists to record that it was wrong, and to record the measurement that showed so. The
probe was written *before* checking the answer, and it prints a verdict rather than a narrative.

## Command

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_lr_schedule.py \
  --optimizer sgd --lr 0.01 --scheduler cosine --epochs 2 --steps 300'
```

The probe builds the loaders exactly the way `AlexNet/train.py` does (same sampler, same
`train_samples_per_epoch`, same transform), attaches a `Callback` that samples
`optimizer.param_groups[0]["lr"]` in `on_before_optimizer_step`, and prints the trajectory. It
writes no weights.

## Raw output, verbatim

```
=== lr trajectory ===
  optimizer=sgd  scheduler=cosine  nominal lr=0.01
  max_epochs=2  steps/epoch=300  classes=64  image_size=227
  optimizer steps recorded : 600
  max / min / final lr     : 1.000000e-02 / 5.000000e-03 / 5.000000e-03
  sampled trajectory:
    step      0: lr=1.000000e-02  (100.00 % of nominal)
    step     50: lr=1.000000e-02  (100.00 % of nominal)
    step    100: lr=1.000000e-02  (100.00 % of nominal)
    step    150: lr=1.000000e-02  (100.00 % of nominal)
    step    200: lr=1.000000e-02  (100.00 % of nominal)
    step    250: lr=1.000000e-02  (100.00 % of nominal)
    step    300: lr=5.000000e-03  ( 50.00 % of nominal)
    step    350: lr=5.000000e-03  ( 50.00 % of nominal)
    step    400: lr=5.000000e-03  ( 50.00 % of nominal)
    step    450: lr=5.000000e-03  ( 50.00 % of nominal)
    step    500: lr=5.000000e-03  ( 50.00 % of nominal)
    step    550: lr=5.000000e-03  ( 50.00 % of nominal)
    step    599: lr=5.000000e-03  ( 50.00 % of nominal)

  steps below 1 % of nominal lr : 0/600 (0.0%)
  mean lr over the run          : 7.500000e-03 (75.0% of nominal)

VERDICT: PASS -- the observed lr stays in a usable range for this configuration.
```

## Reading the numbers

The LR is `1.0e-2` for the whole of epoch 1 and `5.0e-3` for the whole of epoch 2 — a clean
two-level staircase with exactly one transition, at step 300, which is the epoch boundary. That is
`interval="epoch"` behaviour. Lightning is stepping the scheduler **once per epoch**, `T_max=2` is
being honoured as *two epochs*, and the cosine is doing what the code intends.

The "per-step collapse" theory predicted the opposite shape: a smooth decay to `0.0` across the
first 60 steps and then a flat zero for the remaining 540. Zero steps below 1 % of nominal rules
that out directly.

**Why the source reading misled.** `T_max=horizon` with `horizon = trainer.max_epochs` *looks* like
a units mismatch, and the Lightning default really is `interval="step"` for a bare
`{"lr_scheduler": ...}` dict in the general case. What the reading missed is that Lightning
normalises a scheduler that declares itself epoch-wise, and `CosineAnnealingLR` tracks
`last_epoch` in a way that makes the effective interval epoch-aligned here. The lesson is the one
`AGENT_INSTRUCTIONS` §4 already states: a plausible reading of the source is not a measurement. Had
I shipped the "fix the scheduler" change on the strength of the source, I would have altered a
working component and left `P2-25` unexplained.

## Caveats

- This measures the LR the **optimizer sees**, not the gradient magnitudes. A healthy LR is
  compatible with SGD failing for other reasons — see
  [E034](./E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md) and
  [E038](./E038-per-group-lr-sgd-sweep.md).
- `--steps 300 --epochs 2` was chosen so that `steps/epoch` exceeds `T_max`; had I used the default
  full epoch, the staircase would be the same but harder to read against `T_max`.
- The probe's `VERDICT: PASS` means only "the LR is usable", not "the run converges". It ran with
  `limit_val_batches=0` and is not an accuracy measurement.
- `scheduler="step"` and `scheduler="none"` are also selectable; only `cosine` was the suspect, and
  only `cosine` is claimed here.

## Verdict

The cosine schedule is stepped per epoch, holds the nominal LR for the whole of each epoch, and
never collapses. The schedule is **not** why AlexNet's SGD arm sits at chance. `P2-25` stays open
with one fewer candidate.
