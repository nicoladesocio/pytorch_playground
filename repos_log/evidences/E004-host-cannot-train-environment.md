# E004 — Host had no usable training environment

| | |
|---|---|
| **Status** | SUPERSEDED — host-specific, not a repo property |
| **Cards** | (environment; see `P2-19`) |
| **Host** | Windows + WSL2, GTX 960M (`sm_50`), driver 465.89 |
| **First recorded** | [HANDOFF_001 §3d](../handoffs/HANDOFF_001.md) |

## Claim

`torch`, `torchvision`, `lightning`, `pytorch_lightning` were all absent on the host, and the
GTX 960M (`sm_50`, driver 465.89) is too old for current PyTorch CUDA wheels.

The agent deliberately did **not** install a ~2.5 GB CUDA PyTorch stack — it would not have been
usable on that GPU, and the project's real target was the devcontainer.

## Why this file exists

This is the earliest record of a host limitation that later got written into
`AGENT_INSTRUCTIONS.md` §4 and `goal.md` as a repo-wide rule ("GPU is unusable, run on CPU").
That generalisation was wrong and survived three handoffs — see
[E027](./E027-hardware-probe-gpu-ok.md) and card `P2-19`.

**Do not read this file as "the GPU is unusable".** It describes one machine from HANDOFF_001.
Run `python probes/check_hardware.py` on yours.
