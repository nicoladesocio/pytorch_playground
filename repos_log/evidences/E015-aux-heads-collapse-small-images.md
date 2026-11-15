# E015 — GoogLeNet's auxiliary heads collapse below ~128 px

| | |
|---|---|
| **Status** | CONFIRMED (probe-design constraint, not a shipping bug) |
| **Cards** | `P2-16` |
| **Command** | `python probes/check_aux.py` |
| **First recorded** | [HANDOFF_003 §6.4](../handoffs/HANDOFF_003.md) / §3.9 |

## Output

`InceptionOutClassifier` uses `AvgPool2d(kernel_size=5, stride=3)` before its 1×1 conv.

| `--image_size` | aux tap spatial | aux `feature_map_size` |
|---|---|---|
| 96 | 6 | **1** ← crippled |
| 128 | 8 | 2 |
| 160 | 10 | 2 |
| 224 | 14 | 4 (correct) |

At `feature_map_size == 1` the aux head sees a single spatial position, so its gradient is a
global-average signal and it stops acting as a deep-supervision branch.

## Consequences

- **Do not use GoogLeNet probes below 128 px** to judge convergence — you are measuring a crippled
  branch. Probes default to 160 px and warn below 128.
- The production target is 224 px, where the geometry is correct, so this is a probe-design
  constraint rather than a shipping bug.
- A legitimately small GoogLeNet config would need
  `kernel_size=min(5, spatial)` / `stride=min(3, spatial)` or an adaptive pool. Left alone because
  224 px is the only config the final script uses.

This is also why GoogLeNet reaching `acc 1.000` in
[E023](./E023-learning-matrix-both-devices-both-caches.md) at 160 px is meaningful: the aux
branches are contributing there, which they cannot below ~128 px.
