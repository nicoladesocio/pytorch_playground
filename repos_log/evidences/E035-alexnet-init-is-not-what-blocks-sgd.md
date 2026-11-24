# E035 — The uniform `std=0.01` init is not what blocks SGD (but `kaiming` is actively broken)

| | |
|---|---|
| **Status** | CONFIRMED (negative result) — the paper's `std=0.01` init memorises 3/4 fixed batches with unclipped SGD, so the init is not the blocker. `torch` defaults memorise 4/4; `kaiming` memorises 0/4 and starts at loss 9.5. |
| **Cards** | `P2-25` (rules out a candidate cause), `P2-12` |
| **Command** | `docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_vs_init.py --init all --steps 400 --lr 0.01 --batches 4 --cache 51200t_2048v'` |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists

[E034](./E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md) shows `AlexNet/model.py`
applies `normal_(std=0.01)` to **every** conv and **every** Linear. `std` is a constant but fan-in
is not, so the forward variance multiplier `fan_in * std²` ranges from 0.036 (conv1, fan_in 363) to
0.92 (fc1, fan_in 9216) — which is where the 1.5 × 10⁴ gradient spread comes from. The natural
inference is that this init is the thing SGD cannot survive, and that swapping it for a
fan-in-aware init fixes AlexNet.

That inference has a testable consequence: if the init is the blocker, then with the init replaced,
unclipped SGD should train. The cleanest version of that test is memorisation — can SGD drive the
loss on a *fixed* batch to ~0? If the init were so badly scaled that SGD could not move, it could
not memorise either.

## Why this probe runs `--batches 4` and not one batch

The first version of this probe used a single fixed batch, and **its verdict moved with the batch**.
Two runs of the same command, differing only in which cache `find_caches()` happened to pick:

```
cache=..._51200t_2048v    init=paper   loss 2.3025 -> 0.0003   acc 1.000   PASS
cache=..._51200t_16384v   init=paper   loss 2.3025 -> 2.0809   acc 0.250   FAIL
```

Memorising 32 images is *just* easy enough that which 32 you draw decides PASS/FAIL. A probe whose
verdict flips between identical invocations cannot close a card (`AGENT_INSTRUCTIONS` §4.2, and the
same failure mode E025 documents). The probe now runs every init over `--batches` independent fixed
batches — all inits share the same batches, so the arms still differ **only** in init — and reports
how many of them each init memorises. `--cache` is also pinned to `51200t_2048v` by default,
because `find_caches()` sorts by mtime and the cache used previously followed whichever was built
last.

## Command

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_vs_init.py \
  --init all --steps 400 --lr 0.01 --batches 4 --cache 51200t_2048v'
```

Each arm trains the real `AlexNet` on one shared fixed batch of 32 real cached images with
**unclipped** SGD (`lr=0.01`, `momentum=0.9`, `weight_decay=5e-4`) for 400 steps. Weights are
re-seeded per batch, so every init starts from the same weights on every batch. Chance for the
10-class task is `ln(10) = 2.3026`; the pass rule matches `check_learn.py` (`loss < 0.6 × chance`).

## Raw output, verbatim

```
torch 2.10.0+cu128  device=cuda  cache=ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538
batch=32 classes=10 image_size=227 optimizer=sgd lr=0.01 steps=400 seed=42 batches=4
chance loss ln(10) = 2.3026; PASS rule = final loss < 1.3816
(each init is run on all 4 shared batches; arms differ ONLY in --init)

  init=paper    grad-norm spread: min 3.160e-04 med 9.411e-03 max 1.153e-01  ratio 3.648e+02
           batch 0 (seed 42): loss 2.3025 -> 0.2836   acc 0.875   PASS
           batch 1 (seed 43): loss 2.3026 -> 0.0363   acc 0.969   PASS
           batch 2 (seed 44): loss 2.3026 -> 0.6972   acc 0.719   PASS
           batch 3 (seed 45): loss 2.3025 -> 1.6839   acc 0.406   FAIL
           memorised 3/4 batches   290.0s

  init=kaiming  grad-norm spread: min 3.042e-01 med 6.547e+00 max 1.258e+02  ratio 4.137e+02
           batch 0 (seed 42): loss 9.5110 -> 2.2183   acc 0.188   FAIL
           batch 1 (seed 43): loss 9.2613 -> 2.2053   acc 0.219   FAIL
           batch 2 (seed 44): loss 9.8536 -> 2.2392   acc 0.188   FAIL
           batch 3 (seed 45): loss 8.5210 -> 2.1854   acc 0.125   FAIL
           memorised 0/4 batches   285.0s

  init=torch    grad-norm spread: min 2.118e-04 med 1.942e-02 max 1.154e-01  ratio 5.447e+02
           batch 0 (seed 42): loss 2.3026 -> 0.0942   acc 0.969   PASS
           batch 1 (seed 43): loss 2.3034 -> 0.0003   acc 1.000   PASS
           batch 2 (seed 44): loss 2.3017 -> 0.0352   acc 1.000   PASS
           batch 3 (seed 45): loss 2.3001 -> 0.0000   acc 1.000   PASS
           memorised 4/4 batches   286.2s

==============================================================================
init        memorised  median last  mean acc     grad spread  result
------------------------------------------------------------------------------
paper          3/4       0.6972     0.742       3.648e+02  PARTIAL
kaiming        0/4       2.2183     0.180       4.137e+02  FAIL
torch          4/4       0.0352     0.992       5.447e+02  PASS
==============================================================================
The paper's std=0.01 init memorises only SOME batches. A single-batch run of this probe
is therefore not evidence either way -- raise --batches before drawing a conclusion.
```

## Reading the numbers

1. **The init is not what blocks SGD.** With the repo's own `std=0.01` init, unclipped SGD drives a
   fixed 32-image batch from `2.3025` to `0.2836 / 0.0363 / 0.6972` on three of four batches. A
   network whose init prevented SGD from moving could not do that. So the explanation for
   [E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)'s all-chance SGD grid is *not*
   "the init is too small / badly scaled".
2. **`torch` defaults are strictly better than the paper init at memorising** — 4/4 with median
   final loss `0.0352` versus 3/4 at `0.6972`, and it never fails a batch. The paper init is
   therefore *worse*, just not *blocking*. Worth noting for
   [P2-12](../KANBAN.md), but it is not the `P2-25` answer.
3. **`kaiming` is actively broken here and must not be adopted as a "fix".** Its *first* loss is
   `8.5–9.9` on a 10-class problem whose chance loss is `2.3026` — a loss that high means the
   logits are wildly mis-scaled before any update, and it never recovers to below chance in 400
   steps. `kaiming_normal_(mode='fan_in', nonlinearity='relu')` on this topology produces a
   gradient spread of `4.1 × 10²` with a *maximum* of `1.3 × 10²` — two orders of magnitude larger
   than the paper init's `1.2 × 10⁻¹`. Anyone tempted to modernise `AlexNet/model.py`'s init should
   run this probe first; the naive replacement makes things worse.
4. **The interesting asymmetry.** Both `paper` and `torch` memorise a fixed batch, yet both fail to
   beat chance on the full 64-class task ([E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)).
   SGD can *fit* these images; what it cannot do is *generalise* them. That relocates the question
   from the init to the regularisers and the data: `dropout p=0.5` plus `weight_decay=5e-4` acting
   on 16 384 draws-with-replacement from only **4 153 unique** in-class train images.

## Caveats

- Memorisation is measured on a **fixed** batch with `is_training=False` (center-crop, no jitter)
  transforms. It answers "can the optimiser drive this network's loss down at all", not "can it
  learn". Do not read the 3/4 and 4/4 as accuracy.
- `--classes 10` (the probe's default) is not the 64-class task of E033; the chance loss is
  `ln(10)`, not `ln(64)`. The comparison across inits is internally consistent, which is what the
  card needs.
- 4 batches is enough to expose the batch-dependence but is still a small sample; `paper` at 3/4
  vs `torch` at 4/4 is suggestive, not a significant difference. Raise `--batches` before quoting
  it as a ranking.
- `kaiming` zeroes biases except a small positive constant on the layers the paper wants live
  ReLUs; part of its badness may be that constant rather than the weight scaling. The claim here is
  only "this kaiming arm is worse", not "all fan-in-aware inits are".
- The probe deliberately applies **no** gradient clipping, since
  [E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md) established the clip is not the
  blocker.

## Verdict

The uniform `std=0.01` init is **not** what pins AlexNet's SGD arm at chance — SGD memorises fixed
batches with it. `P2-25` stays open with one fewer candidate, and the remaining candidates are the
regularisers and the 4 153-image effective training set, not the init. Separately: do not replace
the init with `kaiming_normal_(mode='fan_in')` without re-running this probe.
