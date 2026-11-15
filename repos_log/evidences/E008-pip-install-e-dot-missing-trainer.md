# E008 — `pip install -e .` does not install the `trainer` CLI

| | |
|---|---|
| **Status** | REPRODUCED → fixed in HANDOFF_003 |
| **Cards** | `P1-7` |
| **Command** | build the wheel, inspect its contents |
| **First recorded** | [HANDOFF_002 §4](../handoffs/HANDOFF_002.md) |

## Evidence

`pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages` listed `AlexNet`, `GoogLeNet`,
`utils`, `custum_datasets`, `trainer_v2` — but **not `trainer/`**. Verified by building the wheel:
`trainer/` is absent from the archive.

Consequence: the one working entrypoint (`python -m trainer`, see
[E001](./E001-trainer-cli-invocation-broken.md)) only worked from a source checkout, never from an
installed package.

## Resolution

`trainer` added to `packages` in HANDOFF_003; verified mechanically by
[E016](./E016-kanban-acceptance-gate-40-of-40.md).
