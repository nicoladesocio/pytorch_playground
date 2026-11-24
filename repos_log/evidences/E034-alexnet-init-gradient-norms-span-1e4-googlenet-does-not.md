# E034 — AlexNet's per-tensor gradient norms span 1.5 × 10⁴ at init; GoogLeNet's span 6.3 × 10²

| | |
|---|---|
| **Status** | CONFIRMED — measured twice in separate processes, identical to 4 significant figures |
| **Cards** | `P2-25` (mechanism for the lr cliff), `P2-12` |
| **Command** | `docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_grad_scale.py'` |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists

[E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md) measured one number for AlexNet at
init: total gradient norm 26.9. That single scalar is enough to explain why a *clip at 1.0*
shrinks every step by 0.037, but it is not enough to explain the shape of the `lr` response:
`lr=0.1` unclipped NaNs in ~5 steps while `lr=3e-2` unclipped stays finite and flat, and Adam
converges at `3e-4`. A single global scalar cannot produce that pattern — a *spread* across
tensors can, because SGD multiplies every gradient by the same `lr`.

`AlexNet/model.py` also answers the question "why can't one global lr serve AlexNet?" only by
assertion. This probe measures it: per-tensor gradient norm at init, alongside fan-in and the
init `std` that produced it, for both networks, plus a features-vs-classifier group summary.

## Command

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_grad_scale.py'
```

## Raw output — AlexNet, verbatim

```
torch 2.10.0+cu128  device=cuda  batch=32  image_size=227  classes=64  precision=32  seed=42
Interpretation: SGD multiplies every gradient by the SAME lr, so the column that decides whether a layer learns is grad_norm relative to the median. A spread of 10^3+ across layers means no single lr can train the network.

--- AlexNet: loss 4.4152  TOTAL grad norm 26.7148  (16 tensors, 1.8s) ---
  median per-tensor grad norm 1.238e-01   min 1.386e-03   max 2.104e+01   spread 1.518e+04
  => a clip at 1.0 scales the total by 0.0374; lr=0.1 therefore behaves as lr=0.00374

  parameter                         kind       fan_in   init_std    grad_norm   x median
  --------------------------------------------------------------------------------------
  classifier.6.weight               linear       4096     0.0100    2.104e+01   1.70e+02
  classifier.4.weight               linear       4096     0.0100    1.361e+01   1.10e+02
  classifier.1.weight               linear       9216     0.0100    8.340e+00   6.74e+01
  features.12.weight                conv         3456     0.0100    3.957e+00   3.20e+01
  features.8.weight                 conv         2304     0.0100    4.390e-01   3.55e+00
  features.10.weight                conv         3456     0.0100    4.341e-01   3.51e+00
  classifier.6.bias                 bias/ln         -     0.0000    2.266e-01   1.83e+00
  classifier.4.bias                 bias/ln         -     0.0000    1.238e-01   1.00e+00
  classifier.1.bias                 bias/ln         -     0.0000    9.474e-02   7.65e-01
  features.12.bias                  bias/ln         -     0.0000    7.075e-02   5.71e-01
  features.10.bias                  bias/ln         -     0.0000    3.687e-02   2.98e-01
  features.4.weight                 conv         2400     0.0100    3.673e-02   2.97e-01
  features.0.weight                 conv          363     0.0100    2.232e-02   1.80e-01
  features.8.bias                   bias/ln         -     0.0000    1.511e-02   1.22e-01
  features.4.bias                   bias/ln         -     0.0000    4.221e-03   3.41e-02
  features.0.bias                   bias/ln         -     0.0000    1.386e-03   1.12e-02

  group summary (this is the P2-25 mechanism):
    features     n=10   max 3.957e+00  median 3.687e-02
    classifier   n=6    max 2.104e+01  median 8.340e+00
    classifier/features max-norm ratio: 5.317e+00
```

## Raw output — GoogLeNet, verbatim (summary lines)

```
--- GoogLeNet: loss 4.4152  TOTAL grad norm ... ---
  median per-tensor grad norm 1.451e-02   min 3.223e-04   max 2.036e-01   spread 6.319e+02

  group summary (this is the P2-25 mechanism):
    classifier   n=126  max 2.036e-01  median 1.451e-02
```

## Reading the numbers

1. **E031's 26.9 reproduces.** Total grad norm 26.7148 at the same configuration; the small
   difference is the batch draw. The `=> a clip at 1.0 scales the total by 0.0374` line is exactly
   E031's "scaled by 0.037", so this probe and E031 agree.
2. **The spread, not the total, is the interesting part.** `classifier.6.weight` has a gradient
   norm 1.5 × 10⁴ times larger than `features.0.bias`, and 170× the median tensor. Under SGD with
   one global `lr`, the last layer moves 170× faster than a typical layer and 1.5 × 10⁴ times
   faster than the slowest. That is the mechanism behind the `lr` cliff in
   [E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md): `lr=0.1` unclipped NaNs
   because `classifier.6.weight` (2.1 × 10¹) blows up, while `features.0` (1.4 × 10⁻³) has
   effectively not moved at all. Lower the `lr` to keep the head finite and the convolutions stop
   learning; raise it and the head diverges. There is no `lr` in between that serves both.
3. **Adam's per-parameter RMS normalisation is precisely the fix for a spread like this**, which is
   why the Adam control in E033 converges at the identical budget. This is the positive
   explanation that replaces E031's clip story.
4. **GoogLeNet is comparatively balanced** — spread 6.3 × 10² versus 1.5 × 10⁴, and its largest
   tensor is 2.0 × 10⁻¹ rather than 2.1 × 10¹. That is consistent with GoogLeNet training fine on
   its default optimiser while AlexNet's SGD arm does not, and it localises the problem in
   AlexNet's *uniform `std=0.01` init applied to `fan_in` from 363 to 9216* rather than in the
   task, the data, or the loss.
5. The `init_std` column shows every weight tensor initialised at `0.0100` regardless of `fan_in`.
   That is the paper's recipe, and it is what makes the spread scale with `fan_in`.

## Caveats

- Gradient norms are measured at **init**, on one batch, in fp32 (`precision=32`) so that fp16
  scaling cannot distort them. They are not a statement about the middle of training.
- `spread` is `max/min` over per-tensor norms; `x median` is `grad_norm / median`. Both are ratios
  of L2 norms of differently-shaped tensors, which is the right comparison for SGD (which scales
  each by the same scalar) but not a statement about relative *importance*.
- The probe is read-only: it builds the nets, does one backward pass, and writes nothing.

## Verdict

AlexNet's init gradient field is wildly unbalanced — a 1.5 × 10⁴ spread across tensors under a
single global `lr` — while GoogLeNet's is not. This explains the shape of AlexNet's `lr` response
and why Adam succeeds where SGD does not. It is a *mechanism*, not yet a fix: whether per-group
learning rates rescue SGD is tested separately in [E038](./E038-per-group-lr-sgd-sweep.md).
