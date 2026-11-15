# E022 — Overfit-one-batch on GPU: both architectures can learn

| | |
|---|---|
| **Status** | CONFIRMED |
| **Cards** | `P2-11` (partial), `P2-23` |
| **Command** | `python probes/check_overfit2.py --device cuda --steps 120` |
| **Host** | RTX A1000 6GB Laptop GPU, `ppg-dev` |
| **First recorded** | [HANDOFF_004 §6.3](../handoffs/HANDOFF_004.md) |

## Method

Synthetic batch at `image_size=160` (chosen so GoogLeNet's aux heads are not crippled — see
[E015](./E015-aux-heads-collapse-small-images.md)), 120 steps.

## Output

| Network | chance loss | first | last | reduction | verdict |
|---|---|---|---|---|---|
| AlexNet | 2.3026 | 2.7690 | 0.2452 | +91.1 % | **LEARNED** |
| GoogLeNet | 3.6841 | 3.6907 | 2.0877 | +43.4 % | **LEARNED** |

Chance thresholds are per-network ([E019](./E019-chance-level-per-network.md)): `ln(K)` for
AlexNet, `1.6·ln(K)` for GoogLeNet's combined loss.

## What this proves, and what it does not

Proves the optimiser and backprop path work on the GPU with real gradients. Uses **synthetic
noise**, so it says nothing about the data path — for that see
[E023](./E023-learning-matrix-both-devices-both-caches.md).

CPU equivalent: [E014](./E014-overfit-one-batch-cpu.md).
