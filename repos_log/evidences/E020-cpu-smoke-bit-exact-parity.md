# E020 — CPU smoke runs reproduce bit-exactly on a second machine

| | |
|---|---|
| **Status** | CONFIRMED — reproduced twice on the second host |
| **Cards** | `P0-5` (CPU half), `P2-19` |
| **Command** | `CUDA_VISIBLE_DEVICES="" python <Model>/train_final.py --smoke` |
| **Host** | RTX A1000 6GB Laptop GPU host, native Ubuntu 22.04, `ppg-dev` container |
| **First recorded** | [HANDOFF_004 §6.1](../handoffs/HANDOFF_004.md) (originals: [HANDOFF_003 §6.8](../handoffs/HANDOFF_003.md)) |

## Method

Run on a **GPU host** with `CUDA_VISIBLE_DEVICES=""` to force the CPU path. This tests two things
at once: cross-machine reproducibility, and that making `--smoke` device-adaptive did not alter
CPU behaviour.

## Output — `AlexNet/train_final.py --smoke`

```
==================== RESULT ====================
  train_loss = 2.5463
   train_acc = 15.6250
    val_loss = 2.4703
     val_acc = 8.3333
      device = cpu
   precision = 32-true
 checkpoints = /workspace/AlexNet/checkpoints
 tensorboard = /workspace/Data/logs
Chance-level accuracy for 10 classes is 10.00%; val_acc must clearly exceed it.
================================================
```

## Output — `GoogLeNet/train_final.py --smoke`

```
==================== RESULT ====================
      train_loss = 3.6667
 train_loss_main = 2.3007
       train_acc = 12.5000
        val_loss = 2.3032
         val_acc = 8.3333
          device = cpu
       precision = 32-true
     checkpoints = /workspace/GoogLeNet/checkpoints
     tensorboard = /workspace/GoogLeNet/logs
Chance-level accuracy for 10 classes is 10.00%; val_acc must clearly exceed it.
================================================
```

## Verdict

**Every number matches HANDOFF_003 §6.8 exactly** — on a different CPU generation, different
kernel, different torch build string. `goal.md`'s reproducibility requirement holds across
machines.

Both land on chance after 2 epochs, which is the expected signature of "correct code, not enough
training", not of a broken network. Compare against the thresholds in
[E019](./E019-chance-level-per-network.md): GoogLeNet's `train_loss 3.6667` vs `1.6·ln(10) = 3.6842`
and `train_loss_main 2.3007` vs `ln(10) = 2.3026` are both exactly at chance.

The `device` and `precision` lines were added in HANDOFF_004 so a handoff can never again be
unable to tell a CPU result from a GPU result — the ambiguity that let a stale "CPU only" belief
survive three handoffs.
