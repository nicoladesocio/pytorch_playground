# E019 — Chance level differs per network: `1.6·ln(K)`, not `ln(K)`

| | |
|---|---|
| **Status** | CONFIRMED (arithmetic, verified against measured first-step losses) |
| **Cards** | `P2-17` |
| **Command** | compare first-step loss against `ln(K)` and `1.6·ln(K)` at K=10 |
| **First recorded** | [HANDOFF_003 §3.10](../handoffs/HANDOFF_003.md) |

## The rule

| Network | Loss being compared | Chance level at K=10 |
|---|---|---|
| AlexNet | `main` | `ln(K)` = **2.3026** |
| GoogLeNet | `main + 0.3·aux1 + 0.3·aux2` | `1.6·ln(K)` = **3.6841** |
| GoogLeNet | `main` only | `ln(K)` = **2.3026** |

GoogLeNet's training loss is `main + 0.3·aux1 + 0.3·aux2`. At initialisation all three heads
output a uniform distribution, so the expected **starting** loss is
`(1 + 2·0.3)·ln(K) = 1.6·ln(K)` — 3.684 for K=10, not 2.303.

## Why it is evidence, not opinion

Measured first-step losses land exactly on these values:
[E014](./E014-overfit-one-batch-cpu.md) reports GoogLeNet step 1 = 3.6900 against 3.6842, and
[E021](./E021-gpu-smoke-runs.md) reports `train_loss 3.6774` with `train_loss_main 2.3035`
against `ln(10) = 2.3026`. Both networks are *at chance*, not broken.

## Cost of getting it wrong

Comparing the combined loss to `ln(K)` reports a healthy network as broken. This mislabelled
GoogLeNet as defective in HANDOFF_003's first pass and again in its reading of the smoke runs,
costing real debugging time on a non-problem.

## Implementation

`chance_loss()` is a shared helper (in `probes/check_overfit2.py`, imported by
`probes/check_learn.py`), and `utils/metrics.py` states the threshold in the RESULT block so a
reader cannot mistake chance for learning. Every convergence assertion must use it.
