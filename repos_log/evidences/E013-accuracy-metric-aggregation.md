# E013 — Accuracy metric: measured Lightning aggregation semantics

| | |
|---|---|
| **Status** | CONFIRMED → fixed in HANDOFF_003 |
| **Cards** | `P2-1` (the `PercentageAccuracy` fix) |
| **Command** | `python probes/check_metric.py` |
| **First recorded** | [HANDOFF_003 §6.2](../handoffs/HANDOFF_003.md) |

## Method

Logs the same ground-truth accuracy (60.0 %) three ways through a real `Trainer.fit()` and prints
the epoch value Lightning reports.

## Output

```
style A: acc = 60.0000 confmat_sum_after_fit=20
style B: acc = 60.0000 confmat_sum_after_fit=-1 (20 => reset per epoch, 60 => not reset)
true global = 60.0000 ; mean-of-batches = 66.6667
```

Style A is `metric.update()` + `self.log(name, metric)`; style B is the same metric driven through
`forward()` and logged as a tensor. Both now agree with the hand-computed global value, and
`confmat_sum_after_fit=20` confirms the state is reset per epoch.

## What was broken

Before the fix, style A printed **6000.0** where the truth was 60.0 — `calculate_accuracy` already
returns a percentage, and it was being logged as if it were a fraction. Note
`mean-of-batches = 66.6667`: that is what the old `calculate_accuracy` + `on_epoch=True` code
computed, and it is why the broken smoke run reported nonsense accuracies.

**Any accuracy number in this repo predating HANDOFF_003 should be treated as untrustworthy.**
