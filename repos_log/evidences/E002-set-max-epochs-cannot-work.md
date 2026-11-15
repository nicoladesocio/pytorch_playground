# E002 — `--set max_epochs=…` cannot work

| | |
|---|---|
| **Status** | REPRODUCED → fixed in HANDOFF_003 |
| **Cards** | `P2-3` |
| **Command** | `python -m trainer train GoogLeNet --set max_epochs=20` |
| **First recorded** | [HANDOFF_001 §3b](../handoffs/HANDOFF_001.md) |

## Claim

`trainer/README.md` advertises `python3 trainer train GoogLeNet --set max_epochs=20`, but
`max_epochs` was a hardcoded `pl.Trainer(...)` kwarg, not an `argparse` option. The CLI emits
`--max_epochs 20` and `argparse` rejects it.

## Evidence

- `max_epochs` hardcoded at `AlexNet/train.py:74` → `100`, `GoogLeNet/train.py:68` → `60`.
- `argparse` exits with `unrecognized arguments: --max_epochs 20`.
- Confirmed by diffing the argparse surface of both train scripts
  (`version_name, num_classes, checkpoint_path, train_samples, val_samples, batch_size`
  [+ `resume` on GoogLeNet]) against the documented override surface.

## Resolution

`--max_epochs`, `--lr`, `--optimizer`, `--scheduler` are real CLI args since HANDOFF_003;
`trainer train <M> --set max_epochs=5` works. Verified mechanically by
[E016](./E016-kanban-acceptance-gate-40-of-40.md).
