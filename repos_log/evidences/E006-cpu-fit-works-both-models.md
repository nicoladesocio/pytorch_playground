# E006 — Both models complete a real `Trainer.fit()` on CPU

| | |
|---|---|
| **Status** | CONFIRMED, and independently reproduced |
| **Cards** | `P0-5` (CPU half), `P2-19` |
| **Command** | `docker exec <container> python <Model>/train.py …` (227 px, CPU, `max_epochs=2`) |
| **Host** | WSL2 container, HANDOFF_002 |
| **First recorded** | [HANDOFF_002 §3](../handoffs/HANDOFF_002.md) |

## Output

```
0 | net | AlexNetLightning | 58.3 M | train
`Trainer.fit` stopped: `max_epochs=2` reached.
  AlexNet    fit() OK  (227px, CPU)

0 | net | GoogLeNetLightning | 10.3 M | train
`Trainer.fit` stopped: `max_epochs=2` reached.
  GoogLeNet  fit() OK  (227px, CPU)
```

## What this settled

No card may be blocked on "needs a CUDA box". The `P2-11` blocker of that name was removed in
HANDOFF_002. Use small `--train_samples` / `--batch_size` / `max_epochs` on CPU.

## Reproduced

The CPU path reproduces **bit-exactly** on a second, different machine (RTX A1000 laptop, native
Ubuntu 22.04) — see [E020](./E020-cpu-smoke-bit-exact-parity.md). Note that this evidence is
about *the pipeline executing*, not about convergence; no run has beaten chance yet (`P2-14`).
