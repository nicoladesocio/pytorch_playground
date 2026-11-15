# E005 — GPU blocked on the HANDOFF_002 host (two independent reasons)

| | |
|---|---|
| **Status** | SUPERSEDED — true of one host, wrongly generalised into a repo rule |
| **Cards** | `P0-5` (the GPU half), `P2-19` |
| **Host** | WSL2 + Docker Desktop, GTX 960M (`sm_50`, 4 GB), driver 465.89 |
| **First recorded** | [HANDOFF_002 §3](../handoffs/HANDOFF_002.md) |

## What was measured on that host

1. **The host driver (465.89) predated WSL2 CUDA GPU passthrough.** Docker Desktop runs
   containers inside WSL2, and WSL2 GPU support requires a much newer driver. The container could
   not even load `libnvidia-ml.so.1`:
   ```
   nvidia-container-cli: initialization error: load library failed:
   /usr/lib/wsl/drivers/nvacui.inf_amd64_.../libnvidia-ml.so.1: file too short
   ```
2. **The GPU is Maxwell `sm_50` (GTX 960M, 4 GB).** Current PyTorch CUDA wheels no longer ship
   `sm_50` kernels. Even with a working driver, prebuilt wheels would not accelerate it.

Reason (1) is fixable by a human updating the NVIDIA driver. Reason (2) is not fixable on that
hardware with modern PyTorch — it would need a source build with `sm_50` enabled.

## Why this file exists (read this before quoting the above)

Both findings were **correct measurements of one machine**. They were then written into
`AGENT_INSTRUCTIONS.md` §4 and `goal.md` as "GPU is unusable / experiments must run on CPU",
which is a claim about the repository. That is the defect `P2-19` fixed.

The generalisable part is reason (2): **check the wheel's arch list against your device.**
`probes/check_hardware.py` does exactly that, because
`torch.cuda.get_device_capability() not in torch.cuda.get_arch_list()` is what actually predicts
usability. See [E027](./E027-hardware-probe-gpu-ok.md) for the same probe returning `GPU_OK` on
different hardware.
