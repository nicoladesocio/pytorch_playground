# E014 — Overfit-one-batch on CPU: both architectures can learn

| | |
|---|---|
| **Status** | CONFIRMED, and reproduced on GPU in [E022](./E022-overfit-one-batch-gpu.md) |
| **Cards** | `P2-11` (partial), `P2-17` |
| **Command** | `python probes/check_overfit2.py --device cpu --steps 120` |
| **First recorded** | [HANDOFF_003 §6.3](../handoffs/HANDOFF_003.md) |

## Method

One **synthetic** batch (`B=32`, `K=10`, 96 px), no DataLoader, no augmentation, Adam `1e-3`,
120 steps. This is the test to run before blaming the data pipeline — "the loss is flat" has at
least four unrelated causes here (data volume, epoch length, learning rate, broken init), and this
cuts the data out entirely.

## Output

```
AlexNet:   chance loss = 2.3026
  step 1    loss 2.5541  grad 2.121e+01  acc 0.125
  step 20   loss 2.3574  grad 8.772e+00  acc 0.188
  step 60   loss 1.9362  grad 2.782e+00  acc 0.250
  step 120  loss 1.1887  grad 7.193e+00  acc 0.469     -> 2.55 -> 1.19  LEARNED

GoogLeNet: chance loss (main) = 2.3026, expected combined chance = 3.6842
  step 1    loss 3.6900  grad 2.400e-01  acc 0.062
  step 40   loss 3.9940  grad 1.423e+01  acc 0.188
  step 120  loss 1.8572  grad 2.480e+01  acc 0.500     -> 3.69 -> 1.86  LEARNED
```

## Reading the GoogLeNet numbers correctly matters

Its loss is `main + 0.3·aux1 + 0.3·aux2`, so at initialisation all three heads are uniform and the
expected **combined** loss is `(1 + 2·0.3)·ln(10) = 3.684` — the observed first-step 3.6900 is
exactly chance, not a bad init. A naive threshold at `ln(K)` mislabels this as "failed to learn";
HANDOFF_003 made that mistake on the first pass and it is recorded here so it is not repeated
(`P2-17`, [E019](./E019-chance-level-per-network.md)).

Both networks reach ~50 % on 32 memorisable samples in 120 steps, so **neither architecture nor
initialisation is the blocker** for convergence.

## Limits of this evidence

Synthetic noise says nothing about the data path. Real images through the real samplers is
[E023](./E023-learning-matrix-both-devices-both-caches.md).
