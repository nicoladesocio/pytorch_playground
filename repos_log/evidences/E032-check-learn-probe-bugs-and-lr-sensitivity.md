# E032 — `check_learn.py` had two RNG bugs; with them fixed, the probe's verdict is dominated by its own default `lr`

| | |
|---|---|
| **Status** | CONFIRMED (bugs + `lr` sensitivity) — supersedes [E029](./E029-augmentation-ab-both-arms-identical-images.md) |
| **Cards** | `P2-15` (reopened — the A/B was invalid), `P2-28` (new), `P2-29` (new), `P2-21` |
| **Command** | see §Commands — every block is copy-pasteable |
| **Host** | `LX0350043`, RTX A1000 6GB Laptop GPU (sm_86), driver 535.309.01, torch 2.10.0+cu128, container `ppg-dev5` from `ppg-gpu:latest` |
| **First recorded** | [HANDOFF_005](../handoffs/HANDOFF_005.md) |

## Why this experiment exists

[E029](./E029-augmentation-ab-both-arms-identical-images.md) closed `P2-15` with "4/4 PASS at
`image_size=227`". While re-running E029's own documented command as a regression check after
editing the probe, **the same command line produced a different failing cell on each invocation**.
A probe whose verdict moves between identical runs cannot close a card, so the probe itself became
the subject.

## Bug 1 — the "fixed" batch was never fixed

`build_batch()` calls `list(sampler)` on a `WeightedRandomSampler`
([`utils/datasets_utils.py:567`](../../utils/datasets_utils.py)). `WeightedRandomSampler` draws
from the **global torch RNG**, and `main()` did not seed before it. So `--seed` pinned the class
selection (a separate `random.Random`) and the network init, but the 32 images themselves were
re-drawn on every invocation. Symptom: identical command, different cell fails. This reads exactly
like GPU nondeterminism and invites a cuDNN rabbit hole.

Fix: `torch.manual_seed(args.seed)` + `torch.cuda.manual_seed_all(args.seed)` immediately after
arg parsing, before anything consumes the RNG.

**Reproducibility proof after the fix** — two concurrent identical runs, byte-identical numbers:

```
rep_a:  AlexNet    cuda  eval   chance 2.3026   2.9965 -> 2.4549   med 2.2906 max 5.432   acc 0.188
rep_b:  AlexNet    cuda  eval   chance 2.3026   2.9965 -> 2.4549   med 2.2906 max 5.432   acc 0.188
```

## Bug 2 — the two arms of the A/B were fed *different* batches

`build_batch()` was called **once per arm** inside the arm loop, so each arm re-drew the sampler.
The eval arm and the train arm therefore memorised two different sets of 32 images while the
docstring claimed they were identical. That is precisely the confound the `--arm` flag was added to
remove, so E029's A/B compared nothing.

Fix: `build_batch()` now returns `(pils, y, K, target_classes)` and is called **once per cache**,
outside the arm loop; `make_arm_transform()` supplies each arm's transform, and both arms re-augment
or reuse the *same* `pils` object. Labels are computed via `class_to_idx[train[i]["label"]]`, which
was checked against `ds[i][1]` on 8 sampled rows and matches exactly — the failures below are not a
mislabelling artefact of the refactor.

## What the corrected probe says

### `--image-size 227`, `--steps 400`, `--arm both`, `--lr 3e-3` (probe default), seed 42

```
=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=eval  (32 images, 10 classes, (227, 227), seed=42) ===
  AlexNet    cuda  eval   chance 2.3026   2.9965 -> 2.4549   med 2.2906 max 5.432   acc 0.188    39.0s   FAIL
  GoogLeNet  cuda  eval   chance 3.6841   3.6843 -> 0.5646   med -0.0000 max 3.554   acc 0.938    47.0s   PASS

=== ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538  arm=train  (32 images, 10 classes, (227, 227), seed=42) ===
  AlexNet    cuda  train  chance 2.3026   2.9963 -> 2.6377   med 2.4571 max 5.086   acc 0.156    92.7s   FAIL
  GoogLeNet  cuda  train  chance 3.6841   3.6847 -> 0.1362   med 0.0012 max 1.121   acc 0.969    99.4s   PASS

OVERALL: 2/4 cell(s) did not beat chance
```

**This contradicts E029's recorded 4/4 PASS on the same command line.** E029's numbers came from a
probe with both bugs present.

### Seed sweep at 227 px, `lr=3e-3` (`--seed 1 2 3`)

Four cells per seed. The `===` header lines were dropped by the `grep` used to capture this log, so
the cache name is not recorded in it — re-run without the grep (or with `--cache 51200`) to confirm
which cache these are from.

```
### seed 1
  AlexNet    eval   2.9012 -> 0.1991   med -0.0000 max 6.368   acc 0.969   PASS
  GoogLeNet  eval   3.6892 -> 6.2738   med 0.0000 max 34.942  acc 0.844   FAIL
  AlexNet    train  2.9012 -> 0.3257   med 0.0023 max 2.170   acc 0.844   PASS
  GoogLeNet  train  3.6892 -> 0.3948   med 0.0000 max 2.347   acc 0.938   PASS
  AlexNet    eval   2.5347 -> 0.0000   ...                       acc 1.000   PASS
  GoogLeNet  eval   3.6882 -> 0.0032   ...                       acc 1.000   PASS
  AlexNet    train  2.5345 -> 1.0442   ...                       acc 0.625   PASS
  GoogLeNet  train  3.6885 -> 0.3729   ...                       acc 0.969   PASS
### seed 2
  AlexNet    eval   2.4951 -> 2.3844   med 2.3269 max 3.515   acc 0.125   FAIL
  GoogLeNet  eval   3.6833 -> 0.0031   ...                       acc 1.000   PASS
  AlexNet    train  2.4951 -> 0.4489   ...                       acc 0.844   PASS
  GoogLeNet  train  3.6836 -> 2.5741   med 0.0000 max 18.737  acc 0.812   FAIL
  (1024 cache: 4/4 PASS)
### seed 3
  AlexNet    eval   2.5719 -> 4.8054   med 2.0712 max 55.403  acc 0.250   FAIL
  GoogLeNet  eval   3.6934 -> 0.4525   ...                       acc 1.000   PASS
  AlexNet    train  2.5717 -> 2.1825   med 2.0105 max 3.675   acc 0.188   FAIL
  GoogLeNet  train  3.6933 -> 0.2242   ...                       acc 0.969   PASS
  (1024 cache: 4/4 PASS)
```

Note `--seed` changes three things at once here: class selection, the batch, and the init. So this
is a sweep over *batches*, which is the right experiment for "does this batch memorise?" and must be
described that way.

### `lr` sweep at 227 px, seed 42 — the decisive result

| `--lr` | cells (both caches, both arms, both nets) |
|---|---|
| `1e-2` | **2/16 PASS** (0/8 at seed 42, 2/8 at seed 3) — most cells pinned at ≈`ln(K)` / `1.6·ln(K)` with `median ≈ mean` and `acc 0.06–0.19`; some driven *above* chance |
| `3e-3` (default) | 4/8 PASS at seed 42 — both AlexNet cells fail on both caches |
| `1e-3` | **16/16 PASS** across seeds 42 and 3 — AlexNet eval `2.9965 → 0.0811 acc 0.969`, AlexNet train `2.9963 → 0.5785 acc 0.812` |

```
### lr 1e-3 seed 42
  AlexNet    cuda  eval   chance 2.3026   2.9965 -> 0.0811   med -0.0000 max 2.597   acc 0.969    39.2s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6843 -> 0.7734   med -0.0000 max 8.509   acc 0.969    47.5s   PASS
  AlexNet    cuda  train  chance 2.3026   2.9963 -> 0.5785   med 0.0102 max 5.539   acc 0.812   105.0s   PASS
  GoogLeNet  cuda  train  chance 3.6841   3.6847 -> 0.1495   med 0.0001 max 1.361   acc 0.969   116.0s   PASS
  AlexNet    cuda  eval   chance 2.3026   3.0148 -> 0.0039   med -0.0000 max 0.072   acc 1.000    38.9s   PASS
  GoogLeNet  cuda  eval   chance 3.6841   3.6854 -> 0.0006   med -0.0000 max 0.000   acc 1.000    47.5s   PASS
  AlexNet    cuda  train  chance 2.3026   3.0145 -> 0.1933   med 0.0000 max 4.958   acc 0.938   101.1s   PASS
  GoogLeNet  cuda  train  chance 3.6841   3.6866 -> 0.1174   med 0.0000 max 2.504   acc 0.969   106.9s   PASS
```

**The same batch, the same seed, the same images that "fail to memorise" at `lr=3e-3` memorise
cleanly at `lr=1e-3`.** The failures are not a property of AlexNet or of the data; they are the
probe's own hyperparameter sitting in a bad spot for a fixed 400-step budget. Raising `lr` to `1e-2`
makes it strictly worse (divergence-like: loss driven *above* chance), so this is not a monotone
"more lr helps" story either.

### `--image-size 160`, `lr=3e-3`, seeds 42/1/2 — 23/24 PASS

Both caches × both arms × both nets × 3 seeds. The only failure was GoogLeNet's train arm on the
sparse 1024 cache at seed 1 (`3.6862 -> 2.6500`, `max 18.279` — one outlier sample dominating the
mean, exactly the statistic E029 introduced `median`/`max` to expose). AlexNet passed all 12 of its
160 px cells.

**Do not read 160 px > 227 px as a resolution finding.** `image_size` is passed to the model
constructor, so it also changes `fc1`'s shape and therefore the parameter-init draw:

```
AlexNet 160  flatten=2304  fc1 in=2304  params=30,010,762
AlexNet 227  flatten=9216  fc1 in=9216  params=58,322,314
```

Resolution, parameter count and init stream are entangled in this probe. Separating them is
`P2-28`.

## What this settles

1. **E029's A/B was invalid** and `P2-15` cannot be closed by it. The two arms did not share a
   batch, and the batch was not pinned by `--seed`.
2. **`check_learn.py`'s pass/fail is not a reliable signal at its default `lr=3e-3`.** A cell flips
   PASS↔FAIL on `lr` alone with everything else fixed. Any card closed on a single `check_learn`
   run is on shaky ground.
3. **AlexNet can memorise 32 real images at 227 px** (`lr=1e-3`: `acc 0.969` eval / `0.812` train).
   The alarming "AlexNet cannot memorise real images" reading of the seed sweep is wrong.
4. **GoogLeNet's occasional train-arm FAILs are outlier-driven means**, not global failure — visible
   only because `median`/`max` are now printed.

## What this does **not** show

- It does **not** re-open `P2-14`/`P2-11`. Those rest on
  [E030](./E030-googlenet-converged-run.md) and
  [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md), which are real `train.py` runs
  against a held-out validation set and never used this probe.
- It does **not** settle `P2-15` in either direction. With a sane `lr` both arms memorise, which
  argues augmentation is not a blocker — but that is a *new* experiment, not E029's.
- It does **not** cover `lr=1e-3` at seeds 1 and 2, nor a `lr`×`steps` grid. `lr=1e-3` is confirmed
  at seeds 42 and 3 (16/16 PASS) and `lr=1e-2` at the same two seeds (3/16). A `lr`×`steps` grid is
  `P2-28`.
- [E023](./E023-learning-matrix-both-devices-both-caches.md)'s "8/8 PASS" was produced by the same
  probe with both RNG bugs present. It is probably fine (it passed, and passing is the easier
  direction) but it is unverified — `P2-29`.

## Commands

```bash
# container (see AGENT_INSTRUCTIONS §7)
docker run -d --name ppg-dev5 --gpus all --entrypoint bash \
  -v "$PWD":/workspace -w /workspace ppg-gpu:latest -c "sleep infinity"

# Bug-1 reproducibility proof
docker exec ppg-dev5 bash -lc 'cd /workspace && for i in a b; do python probes/check_learn.py \
  --steps 400 --arm eval --image-size 227 --seed 42 2>&1 | grep "^  AlexNet"; done'

# 227 px default + seed sweep
docker exec ppg-dev5 bash -lc 'cd /workspace && python probes/check_learn.py --steps 400 --arm both --image-size 227'
docker exec ppg-dev5 bash -lc 'cd /workspace && for s in 1 2 3; do echo "### seed $s"; \
  python probes/check_learn.py --steps 400 --arm both --image-size 227 --seed $s 2>&1 | grep -E "^  (AlexNet|GoogLeNet)"; done'

# 160 px seed sweep
docker exec ppg-dev5 bash -lc 'cd /workspace && for s in 42 1 2; do echo "### seed $s"; \
  python probes/check_learn.py --steps 400 --arm both --image-size 160 --seed $s 2>&1 | grep -E "^  (AlexNet|GoogLeNet)"; done'

# lr sweep (the decisive one)
docker exec ppg-dev5 bash -lc 'cd /workspace && for lr in 1e-2 1e-3; do for s in 42 3; do echo "### lr $lr seed $s"; \
  python probes/check_learn.py --steps 400 --arm both --image-size 227 --seed $s --lr $lr 2>&1 | grep -E "^  (AlexNet|GoogLeNet)"; done; done'

# parameter-count confound
docker exec ppg-dev5 bash -lc 'cd /workspace && python -c "
import importlib.util
s=importlib.util.spec_from_file_location(\"a\",\"AlexNet/model.py\"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
for sz in (160,227):
    n=m.AlexNet(num_classes=10, image_size=sz)
    print(\"AlexNet\",sz,\"flatten=\",n._flatten_features(),\"params=\",sum(p.numel() for p in n.parameters()))"'
```

Each 227 px `--arm both` cell is ~40–120 s on this card; a full seed sweep is ~10 min. A third GPU
process OOMs the 6 GB card — run them serially.
