# E010 — HANDOFF_001's `pytorch_lightning` premise was wrong

| | |
|---|---|
| **Status** | CORRECTED (a claim retracted, not a defect found) |
| **Cards** | `P1-2` |
| **Command** | inspect `lightning` 2.6.6 metadata; import both names |
| **First recorded** | [HANDOFF_002 §4](../handoffs/HANDOFF_002.md) |

## What was claimed, and what is true

HANDOFF_001 asserted that `pytorch_lightning` needed to be declared as a dependency. It does not:
`lightning` 2.6.6 **does** depend on `pytorch_lightning`, so both import fine.

The real problems were different, and are what `P1-2` should have said:

1. Two inconsistent Lightning import styles across the models.
2. `GoogLeNet/model.py:4` uses `from inception import ...` and `GoogLeNet/train.py:4` uses
   `import model` — bare top-level imports that only resolve if `GoogLeNet/` is itself on
   `sys.path`, so `import GoogLeNet.model` fails from the repo root.

## Why this file exists

It is the repo's first recorded instance of the failure mode `P2-19` later named: a plausible
claim written into a shared document without being run. Correcting it cost less than acting on
it would have. Both import problems were fixed in HANDOFF_003 and verified by
[E016](./E016-kanban-acceptance-gate-40-of-40.md).
