# E016 — Kanban acceptance gate: 40/40 mechanical checks

| | |
|---|---|
| **Status** | CONFIRMED |
| **Cards** | the `P1-*` / `P2-*` batch closed in HANDOFF_003 |
| **Command** | `python probes/check_final.py` |
| **First recorded** | [HANDOFF_003 §6.5](../handoffs/HANDOFF_003.md) |

## Method

`probes/check_final.py` re-verifies every card moved to DONE in HANDOFF_003 **mechanically, from
source and from real behaviour** — not from memory. It checks import safety, import style,
portable-path defaults, the `.gitkeep`, the wheel `packages` list, absence of `print()` inside
`training_step`/`validation_step`, seeding, the real CLI flags, the single class-count source,
`--resume`, the checkpoint-helper signatures and `weights_only`, the aux init and aux logging,
`--train_samples_per_epoch` and `drop_last=False` on val, the derived flatten size, and the
registry path round-trip.

## Output

```
40 OK / 0 FAIL
FINAL: ALL PASS
```

## The instructive part: three of its first-run failures were probe bugs

Worth remembering because they are the classic shape of a false positive:

- `256 * 6 * 6` matched inside the explanatory **docstring** of `_flatten_features()`.
- `mode='fan_out'` matched the **Conv2d** branch, which is the correct place for it; only the
  `Linear` branch was in scope.
- The aux metrics are logged as `train_loss_aux1` / `train_loss_aux2`, not `aux1_loss`.

Two were **real** gaps, fixed before that section was written: `--optimizer` / `--scheduler` were
missing on AlexNet, and `--scheduler` was missing on GoogLeNet.

A gate that only ever passes is not a gate. Its value here is that it failed first.
