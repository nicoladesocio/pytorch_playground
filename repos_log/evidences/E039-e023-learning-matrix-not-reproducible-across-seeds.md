# E039 — E023's "8/8 PASS" is not reproducible: 34/36 cells across 3 seeds, and the 2 failures move

| | |
|---|---|
| **Status** | CONFIRMED — E023 is **SUPERSEDED**. Both networks do learn from real images, but `check_learn.py` at its default `lr=3e-3` is not a stable PASS/FAIL test: 34/36 cells pass and the 2 failures land on different caches *and* different devices depending on the seed. |
| **Cards** | `P2-29` (resolves it), `P2-21` (re-established on new numbers), `P2-28` (motivates the retune) |
| **Command** | see §Commands — three copy-pasteable runs, one per seed |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev6` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_006](../handoffs/HANDOFF_006.md) |

## Why this experiment exists

[E023](./E023-learning-matrix-both-devices-both-caches.md) is cited as the evidence for `P2-21`
("both networks learn from real images"). It reports **8/8 PASS**. But it was produced by
`check_learn.py` *before* the two RNG fixes in
[E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md): `main()` did not seed before
`build_batch()`, and `list(sampler)` on a `WeightedRandomSampler` draws from the **global** torch
RNG — so `--seed` never pinned the 32 images. E023's eight cells were eight *different* batches
chosen by an unseeded RNG, and its CPU/GPU columns were not provably the same pixels, which is what
its own Method section claims.

`P2-29`'s acceptance is: re-run with the fixed probe, ≥3 seeds, and either re-confirm E023 with new
numbers or mark it SUPERSEDED.

There is a second reason the old number cannot be compared with a new one: E023 swept **2** caches
and there are now **3** on disk. The third, `51200t_16384v`, is the validation pool this session
built for `P2-27` ([E036](./E036-validation-pool-size-and-accuracy-resolution.md)). So the matrix
grew from 8 cells to 12 per seed.

## Commands

Run **serially**, one seed at a time. See §Caveats for why this matters.

```bash
docker exec ppg-dev6 bash -lc 'cd /workspace && python -u probes/check_learn.py \
  --device both --steps 200 --seed 42'

docker exec ppg-dev6 bash -lc 'cd /workspace && python -u probes/check_learn.py \
  --device both --steps 200 --seed 43'

docker exec ppg-dev6 bash -lc 'cd /workspace && python -u probes/check_learn.py \
  --device both --steps 200 --seed 44'
```

Everything except `--seed` is the probe's default: `lr=3e-3`, `steps=200`, `batch=32`,
`classes=10`, `image_size=160`, `arm=eval`. ~11 min per seed.

## Result

| Seed | Cells | Result |
|---|---|---|
| 42 | **10/12** | 2 FAIL — `AlexNet/cpu/16384v` and `AlexNet/cuda/1024t_256v` |
| 43 | **12/12** | all PASS |
| 44 | **12/12** | all PASS |
| **total** | **34/36** | |

Split by network, which is where the signal is:

| Network | Cells | PASS |
|---|---|---|
| GoogLeNet | 18/18 | **100 %** |
| AlexNet | 16/18 | 89 % — and both failures are AlexNet |

The two failures are **not** the same cell. Seed 42 failed one cell on CPU with the *large* cache and
one cell on GPU with the *small* cache. Neither cell failed on seeds 43 or 44.

## Raw output, verbatim — seed 42 (the seed that fails)

```
torch 2.10.0+cu128   cuda available: True   gpu: NVIDIA RTX A1000 6GB Laptop GPU
devices: ['cpu', 'cuda']
caches:  ['ILSVRC__imagenet-1k_51200t_16384v_auto_cf2d39fe2ecf', 'ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538', 'ILSVRC__imagenet-1k_1024t_256v_auto_bdedb67a1a9e']
Selected 10 classes; per-class cached counts: min=68, median=69, max=73

=== ILSVRC__imagenet-1k_51200t_16384v_auto_cf2d39fe2ecf  arm=eval  (32 images, 10 classes, (160, 160), seed=42) ===
  AlexNet    cpu   eval   chance 2.3026   2.4999 -> 2.3657   med 2.3944 max 4.348   acc 0.250   109.6s   FAIL
  AlexNet    cuda  eval   chance 2.3026   2.4424 -> 0.5841   med 0.0013 max 7.786   acc 0.844     9.8s   PASS
  GoogLeNet  cpu   eval   chance 3.6841   3.6925 -> 1.4862   med -0.0000 max 21.025   acc 0.875   253.1s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6949 -> 0.0388   med -0.0000 max 0.476   acc 1.000    13.5s   PASS
Selected 10 classes; per-class cached counts: min=67, median=68, max=73

=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=eval  (32 images, 10 classes, (160, 160), seed=42) ===
  AlexNet    cpu   eval   chance 2.3026   2.5571 -> 0.3341   med 0.0000 max 2.119   acc 0.875   107.1s   PASS
  AlexNet    cuda  eval   chance 2.3026   2.4813 -> 0.5896   med 0.0002 max 6.552   acc 0.844     9.5s   PASS
  GoogLeNet  cpu   eval   chance 3.6841   3.6872 -> 0.7197   med 0.0000 max 19.782   acc 0.906   228.2s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6867 -> 0.0025   med -0.0000 max 0.009   acc 1.000    13.4s   PASS
Selected 10 classes; per-class cached counts: min=3, median=4, max=5

=== ILSVRC__imagenet-1k_1024t_256v_auto_bdedb67a1a9e  arm=eval  (32 images, 10 classes, (160, 160), seed=42) ===
  AlexNet    cpu   eval   chance 2.3026   2.5732 -> 0.1431   med 0.0050 max 3.766   acc 0.969   106.7s   PASS
  AlexNet    cuda  eval   chance 2.3026   2.4351 -> 2.2985   med 2.2278 max 3.852   acc 0.094     9.5s   FAIL
  GoogLeNet  cpu   eval   chance 3.6841   3.6872 -> 0.1646   med -0.0000 max 0.280   acc 1.000   252.4s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6883 -> 0.1525   med 0.0000 max 4.210   acc 0.969    13.4s   PASS
```

```
OVERALL: 2/12 cell(s) did not beat chance: AlexNet/cpu/eval/ILSVRC__imagenet-1k_51200t_16384v_auto_cf2d39fe2ecf, AlexNet/cuda/eval/ILSVRC__imagenet-1k_1024t_256v_auto_bdedb67a1a9e
```

Read the two failures carefully, because they are not "the network is broken":

- `AlexNet cpu 16384v`: `2.4999 -> 2.3657`, **med 2.3944**. The median is at chance, so the loss
  never came down — this is a genuine failure to fit, not a noisy final reading.
- `AlexNet cuda 1024t_256v`: `2.4351 -> 2.2985`, **med 2.2278**, `acc 0.094`. Same shape: pinned at
  `ln(10) = 2.3026`.

Both are *stuck*, not *slow*. And on seeds 43 and 44 the identical cells reach `med 0.0000`.

## Raw output, verbatim — seeds 43 and 44 (summary tables)

Both seeds produced a clean sweep. Seed 43:

```
model      device  arm    cache                                   result
--------------------------------------------------------------------------------------------
AlexNet    cpu     eval   ILSVRC__imagenet-1k_51200t_16384v_auto_ PASS
AlexNet    cuda    eval   ILSVRC__imagenet-1k_51200t_16384v_auto_ PASS
GoogLeNet  cpu     eval   ILSVRC__imagenet-1k_51200t_16384v_auto_ PASS
GoogLeNet  cuda    eval   ILSVRC__imagenet-1k_51200t_16384v_auto_ PASS
AlexNet    cpu     eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
AlexNet    cuda    eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
GoogLeNet  cpu     eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
GoogLeNet  cuda    eval   ILSVRC__imagenet-1k_51200t_2048v_auto_c PASS
AlexNet    cpu     eval   ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS
AlexNet    cuda    eval   ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS
GoogLeNet  cpu     eval   ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS
GoogLeNet  cuda    eval   ILSVRC__imagenet-1k_1024t_256v_auto_bde PASS

OVERALL: all 12 cells learned -- both networks learn from real images on every device and arm tested, on every cached subset.
```

Seed 44 produced the identical table and the identical `OVERALL` line.

## What this settles

1. **E023 is SUPERSEDED, not re-confirmed.** Its headline "8/8 PASS" cannot be reproduced as a
   property of the code, because no such property exists at `lr=3e-3`. The correct claim is
   "**34/36 cells pass across 3 seeds**", and it must be stated with the seed list.
2. **The underlying claim of `P2-21` survives.** Both networks really do learn from real images
   through the real `choose_target_classes → create_subset_sampler → ImageNetSubsetDataset` path.
   GoogLeNet is 18/18. AlexNet is 16/18 and the 2 misses are a learning-rate artefact, not a data
   or architecture fault — the same cell passes on the other two seeds.
3. **`check_learn.py`'s PASS/FAIL is not a gate.** This is E032's finding, now reproduced on a
   36-cell matrix instead of a single batch: at `lr=3e-3` the verdict flips on the seed. A single
   `check_learn` run must never close a card.
4. **GoogLeNet is the robust one, AlexNet the fragile one** — the same asymmetry
   [E034](./E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md) explains from the init
   gradient spread (1.5 × 10⁴ vs 1.2 × 10²).

## Caveats

- **Run the seeds serially, not concurrently.** My first attempt ran all three at once and the
  `AlexNet cpu` cell took **666 s** instead of 111 s — 20 cores shared three ways. That contention
  was corrupting the CPU column, so I killed all three and re-ran serially. The serial run
  reproduced the same two failures at 111 s, which is what licenses the conclusion that the failures
  are real rather than an artefact of load. **A CPU cell's wall time is a sanity check on the
  whole run's validity.**
- **The matrix is 12 cells, not 8.** E023's 8 was 2 networks × 2 devices × 2 caches; there are now
  3 caches because `P2-27` added `51200t_16384v`. Do not read "12" as a regression.
- **`1024t_256v` is a thin cache**: `min=3, median=4, max=5` images per class, so its 32-image batch
  is drawn from a pool of ~40. It is the hardest cell to fit and it is where seed 42's GPU failure
  landed.
- **CPU and GPU final losses differ** (seed 42, `2048v`: cpu 0.3341 vs cuda 0.5896). Expected, not
  drift — identical pixels, but fp32 reduction order differs between cuDNN and oneDNN. Both clear the
  bar.
- **These are memorisation tests on 32 images, not convergence.** Beating chance over a *dataset* is
  a different question and is what `P2-14`/`P2-11` settled with
  [E030](./E030-googlenet-converged-run.md)/[E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md).
- **`arm=eval` only.** This says nothing about `P2-15` (does augmentation block memorisation?), which
  needs `--arm both`.

## Verdict

**`P2-29` resolved: E023 is SUPERSEDED.** Replace its "8/8 PASS" with "34/36 cells across seeds
42/43/44; GoogLeNet 18/18, AlexNet 16/18". The claim that both networks learn from real images
stands; the claim that `check_learn.py` can *certify* it at `lr=3e-3` does not. The fix is `P2-28` —
retune the default `lr`, which is what the `lr=1e-3` replication in
E040 (`E040-check-learn-default-lr-retune.md`, written alongside the `--lr` default change)
addresses.
