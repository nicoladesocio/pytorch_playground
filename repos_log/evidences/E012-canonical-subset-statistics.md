# E012 — Canonical 51200/2048 subset: statistics and the cache-key/seed gotcha

| | |
|---|---|
| **Status** | CONFIRMED (statistics measured under `seed=1337`; see the gotcha) |
| **Cards** | `P1-8`, `P3-14` |
| **Command** | `python probes/check_big.py` |
| **First recorded** | [HANDOFF_003 §6.1](../handoffs/HANDOFF_003.md) |

## Output

Parquet download + `save_to_disk` took **759 s** (~13 min); reload from cache is **16.5 s** and
needs no network.

```
Loading cached subset of ILSVRC/imagenet-1k from
  Data/datasets/ILSVRC__imagenet-1k_51200t_2048v_auto_2ecb016d7b0d
Train dataset size: 51200
Validation dataset size: 2048
elapsed=16.5s train=51200 val=2048
train classes: 1000
val classes: 883
per-class train min/p25/med/p75/max: 22 47 51 56 73
per-class val   min/p25/med/p75/max: 1 1 2 3 10
Selected 10 classes; per-class cached counts: min=68, median=70, max=73
K=10:  in-class train=701  val=23  classes=[45, 103, 219, 246, 556, 645, 801, 853]...
Selected 64 classes; per-class cached counts: min=62, median=66, max=73
K=64:  in-class train=4210 val=153 classes=[3, 10, 18, 26, 35, 36, 45, 73]...
Selected 128 classes; per-class cached counts: min=59, median=62, max=73
K=128: in-class train=8084 val=303 classes=[3, 10, 18, 22, 26, 35, 36, 45]...
sampler len: 701
DONE
```

## What these numbers mean

- **All 1000 train classes survive** the 51200-sample cut and the per-class distribution is tight
  (22–73, median 51). The subset is *not* pathologically skewed, so `choose_target_classes` has
  plenty of freedom.
- **Validation is the thin side.** 883/1000 classes appear at all and the median class has **1**
  val image (p75 = 3). At `K=64` the whole validation set is **153 images** — one per class on
  average. `val_acc` there has ~0.65 % resolution and ±several-percent noise, so **do not** read a
  1–2 % `val_acc` difference at `K=64` as signal. Raise `val_samples`, or judge convergence on
  `train_loss` plus a larger val split.
- **`K=64` gives 4210 in-class train images** = 44 batches at `batch_size=96`. A real epoch, not
  the 1-batch-per-epoch trap.
- **`sampler len: 701` at `K=10`** confirms `create_subset_sampler` returns exactly one pass over
  the in-class samples with no repeats (`num_samples=None` → `num_in_class`).

## Cache-key / seed gotcha (cost a wasted 5.7 GB build)

`seed` is part of `_cache_key`, and `train_final.py` bakes `"seed": 42`, which hashes to
`ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538`. The statistics above were produced under
`seed=1337` → `..._2ecb016d7b0d`, a **different directory**. `probes/check_big.py` now pins
`SEED = 42` to match the deliverable. Build a subset with any other seed and `train_final.py` will
not find it — it will start a 13-minute download. Check the directory name against
`_cache_key(dataset_name, 51200, 2048, None, 42)` before assuming a cache is warm.

The `seed=42` cache itself was built and verified in [E024](./E024-canonical-subset-correct-cache-key.md).

## `save_to_disk` memory

The first attempt at the `seed=42` subset was **OOM-killed at 2/12 shards** inside the 8.25 GB
container cap; the `seed=1337` build completed all 12 shards (re-run after earlier parts were
already on disk, peak RSS under the cap). The round-trip is *borderline*, not impossible, at this
size — hence `P3-14` (let the cache be the part files, skip `save_to_disk`).
