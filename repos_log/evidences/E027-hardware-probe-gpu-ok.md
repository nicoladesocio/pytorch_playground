# E027 — Hardware probe verdict on the RTX A1000 host: `GPU_OK`

| | |
|---|---|
| **Status** | CONFIRMED — PASS=7 FAIL=0 WARN=0 |
| **Cards** | `P2-19` |
| **Command** | `python probes/check_hardware.py` |
| **Host** | RTX A1000 6GB Laptop GPU, driver 535.309.01, native Ubuntu 22.04, `ppg-dev` |
| **First recorded** | [HANDOFF_004 §6.8](../handoffs/HANDOFF_004.md) |

## Output

```
[PASS] nvidia-smi — NVIDIA RTX A1000 6GB Laptop GPU, driver 535.309.01, 6144 MiB
[PASS] driver version — 535.309.01 supports the CUDA 12.x runtime
[PASS] /dev/shm — 8.6 GB
[INFO] CPUs visible — 20
[INFO] torch — 2.10.0+cu128 (CUDA built 12.8)
[PASS] torch.cuda — 1 device(s): NVIDIA RTX A1000 6GB Laptop GPU, sm_86, 6.2 GB
[PASS] wheel supports device arch — sm_86 in ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
[INFO] mixed precision — bf16=True (needs sm_80+)
[PASS] AlexNet gpu fwd/bwd — loss=3.0174 grads=16/16 0.47s (fp32, B=4, 160px)
[PASS] GoogLeNet gpu fwd/bwd — loss=3.6716 grads=126/126 0.37s (fp32, B=4, 160px)

PASS=7  FAIL=0  WARN=0
VERDICT: GPU_OK
```

## Why this probe exists instead of a paragraph of prose

`AGENT_INSTRUCTIONS.md` §4 and `goal.md` both asserted "GPU is unusable, experiments must run on
CPU". Both were written from **one** WSL2 host (driver 465.89, GTX 960M `sm_50`) — see
[E005](./E005-gpu-blocked-on-one-host.md) — and recorded as repo-wide rules. False on the next
host, and it survived three handoffs unchallenged.

The rule is now **"measure, do not trust this document"**: run the probe first, act on its
verdict (`GPU_OK` / `CPU_ONLY` / `GPU_BROKEN` / `HOST_ONLY`).

## Two design decisions worth copying

1. **The arch-list check is the one that predicts usability.**
   `torch.cuda.get_device_capability() not in torch.cuda.get_arch_list()` is exactly why the 960M
   died — `sm_50` was dropped from modern wheels. That is the generalisable lesson from the old
   failure, not "CUDA is broken".
2. **A matmul smoke test is not evidence of a working GPU.** Driver/wheel mismatches commonly pass
   cuBLAS matmul and die on the first cuDNN conv. The probe therefore runs a real forward +
   backward through **both repo networks** before it will say `GPU_OK`.

## For the next agent on new hardware

Add your row to the per-host table in `AGENT_INSTRUCTIONS.md` §6. Do not rewrite someone else's,
and do not promote your host's verdict into a rule.
