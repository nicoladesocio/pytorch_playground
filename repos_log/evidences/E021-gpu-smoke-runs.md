# E021 — GPU smoke runs: mixed-precision path executes on both models

| | |
|---|---|
| **Status** | CONFIRMED (executes cleanly; **not** evidence of convergence) |
| **Cards** | `P3-13` (plumbing half), `P2-22` |
| **Command** | `python <Model>/train_final.py --smoke` |
| **Host** | RTX A1000 6GB Laptop GPU, driver 535.309.01, torch 2.10.0+cu128, `ppg-dev` |
| **First recorded** | [HANDOFF_004 §6.2](../handoffs/HANDOFF_004.md) |

## Output — `AlexNet/train_final.py --smoke`, exit 0, no NaN

```
  train_loss = 2.6044
   train_acc = 6.2500
    val_loss = 2.4137
     val_acc = 8.3333
      device = cuda:0 (NVIDIA RTX A1000 6GB Laptop GPU)
   precision = 16-mixed
```

## Output — `GoogLeNet/train_final.py --smoke`, exit 0, no NaN

```
      train_loss = 3.6774
 train_loss_main = 2.3035
       train_acc = 9.3750
        val_loss = 2.3034
         val_acc = 8.3333
          device = cuda:0
       precision = 16-mixed
```

## Reading these against the right thresholds

GoogLeNet's `train_loss 3.6774` is **at chance** (`1.6·ln 10 = 3.6841`,
[E019](./E019-chance-level-per-network.md)), not broken. Its `train_loss_main 2.3035` against
`ln 10 = 2.3026` confirms the main head is also exactly at chance.

These are 2-epoch / 1024-sample runs. They prove the mixed-precision GPU path **executes**; they
say nothing about whether the recipe converges (`P2-14` remains open).

## Reproduced

Re-running the AlexNet smoke after the argument-contract change gave byte-identical numbers
(`train_loss = 2.6044`, `train_acc = 6.2500`, `val_loss = 2.4137`, `val_acc = 8.3333`) — the third
independent reproduction of a smoke RESULT block in this repo.
