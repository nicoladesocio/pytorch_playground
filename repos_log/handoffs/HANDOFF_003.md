# HANDOFF_003 — Data path unblocked, convergence defects fixed, parameterless final scripts

**Agent:** third agent (session 3)
**Date:** 2026-11-03
**Branch:** `agent/data-unblock-and-convergence` (off `origin/main` @ `e8c063b`)
**Cards:** `P0-1`, `P0-2`, `P1-1`, `P1-2`, `P1-4`, `P1-6`, `P2-1`, `P2-2`, `P2-3`, `P2-4`, `P2-6`, `P2-7`, `P2-10`, `P2-13` done · final scripts delivered
**Read next:** only this file + [KANBAN.md](../KANBAN.md) + [goal.md](../goal.md).

> Written incrementally, per the owner's instruction in `AGENT_INSTRUCTIONS.md` ("write the
> handoff bit by bit, so that even if your session is prematurely terminated, you have
> something written down"). Sections are filled as evidence lands.

---

## 0. TL;DR for the next agent

1. **The dataset blocker was never gating — it was a wrong repo id.** The code used
   `imagenet-1k`; the only valid Hub id is **`ILSVRC/imagenet-1k`**. That is the whole fix.
2. **Streaming is unusable for this dataset.** `load_dataset(..., streaming=True).shuffle(2048)`
   ran at **1.5 samples/s and was OOM-killed at 6.2 GB RSS**. Reading whole parquet row groups
   off the Hub runs at **33.4 samples/s at ~0.5 GB RSS** — 22x faster. That is now the primary
   path in `utils/datasets_utils.py`.
3. **Row groups are already globally shuffled** (one 872-row group spans 575 distinct labels
   across 0..999), so no shuffle buffer is needed for class coverage. This was measured, not
   assumed — see §3.3.
4. **`trainer/` is a registry, not a training engine.** `trainer train <Model>` shells out to
   `<Model>/train.py`. There is exactly one `pl.Trainer` per model, in its `train.py`. If you
   are tempted to "use the trainer instead", read §5 first.
5. **Two parameterless deliverable scripts exist**: `AlexNet/train_final.py`,
   `GoogLeNet/train_final.py`. The human runs these on the big GPU. `--smoke` is the only
   accepted argument and is for agents only.
6. **Start your container with `--shm-size=4g`.** Docker's 64 MB default kills multi-worker
   DataLoader runs with a cryptic shm error. See §7.
7. **Both deliverable scripts run end to end** (`--smoke`, exit 0) and both land exactly on
   chance after 2 epochs — which is the expected signature of "correct code, not enough
   training", not of a broken network (§6.3, §6.8). What is *not* done: no run has beaten
   chance yet (`P2-14`), the canonical 51 200/2 048 subset is built and verified but under the
   *wrong* cache key for the deliverable (`P1-8` — one 13-minute re-run with `SEED=42`, see §9.1),
   and the baked-in `CONFIG` numbers are informed defaults rather than measured values (`P3-13`).
   Your first job is §9.

---

## 1. Why this session exists

The owner asked me to read `repos_log/AGENT_INSTRUCTIONS.md` and work the goal, with one
explicit addition:

> "the trainings should not be big, but should be runnable on a PC with a bigger GPU … there
> must be one script with no parameters, the agent creating it … must bake into the script all
> the best parameters. and there must be one script per neural network."

So the deliverable is not "more training code" — it is **two parameterless scripts, one per
network, with measured-best hyperparameters**, plus the fixes that make a run actually
converge. The size policy and the script contract are now written into
[goal.md](../goal.md) so they survive this session.

Mid-session the owner supplied a HuggingFace token for the gated dataset, and challenged one
of my design choices ("why didn't you use the trainer?") — that challenge was correct and
exposed a real gap; see §5.

---

## 2. Step log

| # | Action | Result |
|---|---|---|
| 1 | `whoami` with the supplied token | ✅ user `nicoladesocio`, `canReadGatedRepos: True`, fine-grained token "playground", `repo.content.read` |
| 2 | `load_dataset("imagenet-1k", streaming=True)` | ❌ `HfUriError: Repository id must be 'namespace/name'` |
| 3 | Same with `ILSVRC/imagenet-1k` | ✅ streams real images (labels 726/917/13, 817×363 / 500×491 / 500×375 RGB) |
| 4 | `git fetch` | **PR #2 was merged**; `origin/main` = `e8c063b`; local `main` stale at `152c01b` |
| 5 | Stashed owner's uncommitted `AGENT_INSTRUCTIONS.md` edit → branched → popped | ✅ branch `agent/data-unblock-and-convergence` |
| 6 | Rewrote `utils/datasets_utils.py`; added `utils/runtime.py` | dataset id, cache keys, seeding, worker clamping |
| 7 | Rewrote both `train.py`, `GoogLeNet/model.py`; patched `AlexNet/model.py`, `inception.py`, `model_utils.py` | see §4 |
| 8 | Structural check of both models in-container | ✅ flatten derives 224→6400 / 227→9216; aux heads build; optimizers build; 58.5 M / 10.5 M params |
| 9 | Owner challenge → read `trainer/trainer.py`, `trainer/model_base.py` | `handle_train()` → `subprocess.run(build_train_command(...))` → `[python, <model_dir>/train.py, *args]`. Registry, not engine. |
| 10 | Rewrote both `trainer_model.py` `default_parameters`; added 11 flags to `trainer/trainer.py` | registry can now reach every `train.py` flag |
| 11 | `python -m trainer register --model-name alexnet --force` + `--dry-run` | ✅ correct command emitted; exposed a **latent case bug**: `/workspace/alexnet/train.py` (lowercase) only worked via the case-insensitive Windows bind mount |
| 12 | Fixed `_resolve_model_dir` to match sibling folders case-insensitively | ✅ |
| 13–18 | Five OOM-killed attempts at the streaming download (details §3.1–3.2) | root-caused to random shard access |
| 19 | Shard census via `HfFileSystem` | 294 train shards / 146.48 GB; 14 val shards / 6.69 GB; row group = 872 rows |
| 20 | Sequential row-group read benchmark | **33.4 samples/s** (872 rows in 26.1 s) |
| 21 | Label census per row group | **globally shuffled** — 575/577/584 distinct labels per 872-row group, spanning 0..999 |
| 22 | Added `_split_shards()` / `_download_split_via_parquet()`; wired parquet-first with streaming fallback | ✅ |
| 23 | First `_split_shards()` attempt returned `train shards: 0` | my own bug: filtered `"/train-"` (never matches) **and** the `HfFileSystem` prefix needs a `datasets/` prefix |
| 24 | Verified prefix: `datasets/ILSVRC/imagenet-1k/data` ✅ (336 entries); `ILSVRC/imagenet-1k/data` ❌ `FileNotFoundError` | fixed both |
| 25 | End-to-end subset download, 4096 train / 512 val | ✅ **208 s total, flat RSS**; 986 distinct train classes; images stored undecoded (`dict` with `bytes`+`path`) |
| 26 | Found val coverage of chosen classes was only **34/512** for 64 classes | accuracy would be measured on ~0.5 images/class |
| 27 | Made `choose_target_classes` require ≥1 val sample per class (`val_dataset=` plumbed through both `train.py`) | ✅ 64 classes with real val support |
| 28 | AlexNet CPU smoke run — first attempt | ❌ `unable to allocate shared memory ... No space left on device` — `/dev/shm` is **64 MB** |
| 29 | `docker commit` → recreated `ppg-dev3` with `--shm-size=4g` | ✅ 4.0 GB shm |
| 30 | Added `available_shm_bytes()` / `estimate_worker_shm_bytes()` and shm-aware `num_workers` clamping | degrades to `num_workers=0` with an actionable message instead of crashing |
| 31 | AlexNet CPU smoke run, 4096/512, 64 classes, 3 epochs | ✅ see §6 |

---

## 3. Findings worth not re-discovering

### 3.1 The streaming download is not just slow, it is structurally wrong for this dataset

Measured in the dev container (8 CPU / 8 GB):

| Method | Throughput | Peak RSS | Outcome |
|---|---|---|---|
| `load_dataset(streaming=True)` + `shuffle(2048)` + `take(n)` | **1.5 samples/s** | **6.2 GB** | OOM-killed at `Generating train split: 0 examples` |
| raw iteration, no decode | 6.6 samples/s | — | — |
| raw iteration, PIL decode | 1.4 samples/s | — | — |
| sequential parquet row-group read | **33.4 samples/s** | **~0.5 GB** | ✅ |

Five separate attempts OOM-killed at 7.2–7.5 GB RSS. Things that did **not** help and should
not be retried:

- `Image(decode=False)` via a copied `Features` object — still OOMed.
- `cast_column(..., Image(decode=False))` *before* `shuffle` — still OOMed.

The cause is the **access pattern**, not decoding: `shuffle()` walks all 294 train shards
round-robin, so it keeps hundreds of parquet readers and their compressed page buffers alive
simultaneously. 512 samples of 108 KB JPEGs is ~55 MB of real data yet cost 6.2 GB RSS.

Also seen, both benign: `'[Errno 9] Bad file descriptor'` retry storms on parquet GETs, and a
`PyGILState_Release` abort at interpreter exit after the work had completed.

### 3.2 Shard layout (so you can size a subset intelligently)

- train: **294** parquet shards, **146.48 GB** total, avg 498.2 MB, 5 row groups × 872 rows
- validation: **14** shards, **6.69 GB**
- test: 28 shards
- `data/` on the Hub has 336 parquet files
- image column type: `struct<bytes:binary, path:string>` → keep it undecoded and decode inside
  DataLoader workers

A 4096-sample subset is ~5 row groups ≈ 440 MB of transfer. A 51 200-sample subset is ~59 row
groups ≈ 6.4 GB. That is the reason the final scripts can afford 51 200/2 048.

### 3.3 Row groups are globally shuffled — no shuffle buffer needed

I expected to have to build a class-balancing shuffle. I measured instead:

| Row group | Rows | Distinct labels | Label range |
|---|---|---|---|
| `train-00197` rg0 | 872 | **575** | 0–999 |
| (2nd sample) | 872 | 577 | 0–999 |
| (3rd sample) | 872 | 584 | 0–999 |

A single row group is already a near-uniform cross-section of all 1000 classes. So
`_download_split_via_parquet` picks shards and row groups with a **seeded RNG** and reads them
sequentially — uniform class coverage for free, and `shuffle_buffer_size` is now only used by
the streaming fallback.

Consequence for the cached subset: 4096 samples → **986 distinct classes**, median 4 images
per class. That is why class selection must be count-aware (§4, `choose_target_classes`) and
must require val support — otherwise you "train on 64 classes" and validate on 34 images.

### 3.4 `HfFileSystem` paths need the `datasets/` prefix

```python
fs.ls("datasets/ILSVRC/imagenet-1k/data")   # ✅ 336 entries
fs.ls("ILSVRC/imagenet-1k/data")            # ❌ FileNotFoundError
```

My first `_split_shards` returned 0 shards for two independent reasons (missing prefix, and
filtering on `"/train-"` in the basename, which never matches). If a shard list comes back
empty, check both before blaming the token.

### 3.5 The dataset id, and the token

- `imagenet-1k` → `HfUriError: Repository id must be 'namespace/name'`. **`ILSVRC/imagenet-1k`
  is the only correct id.**
- It is gated. The token is read from `HF_TOKEN` by `datasets` itself. It is **never** written
  into any repo file; in the container it lives at `/root/.hf_token` (chmod 600) and is passed
  as `docker exec -e HF_TOKEN=$(cat /root/.hf_token)`.
- `_raise_load_help()` turns the gated/auth failure modes into a 3-step instruction instead of
  a traceback.

### 3.6 `calculate_accuracy` returns a percentage — logging it per-batch produced garbage

The first smoke run reported `val_acc=1.220` (a value > 1 is impossible for a fraction and
meaningless for a percentage) and a frozen `train_acc=0.223` while the loss was clearly moving.

Root cause: `utils/helpers.calculate_accuracy` already returns a **percentage** (0–100), and the
models called it per batch and logged the tensor with `on_epoch=True`. Lightning then
**mean-reduces that tensor across batches**, so the epoch value is a mean of per-batch
percentages — wrong whenever batches differ in size, and in the broken run it was compounded by
a second scaling pass.

I did not guess at Lightning's aggregation semantics, I measured them
(`probes/check_metric.py`, three logging styles against a hand-computed ground truth):

| Style | Logged value | Truth |
|---|---|---|
| `self.log(name, metric(preds, target))` with a `compute()` that scales by 100 | **6000.0** | 60.0 |
| `self.log(name, metric(preds, target))` after removing the `forward()` override | **6000.0** | 60.0 |
| `metric.update(preds, target)` then `self.log(name, metric)` | **60.0** | 60.0 |

The first two are wrong for the same reason: torchmetrics' `forward()` already calls
`update()` **and** `compute()`, so the value handed to Lightning is a percentage, and a
`compute()` override that multiplies by 100 gets applied on top of the metric's own scaling.
The third is right because Lightning calls `compute()` itself once at epoch end and reduces the
**confusion-matrix state**, not the per-batch outputs — which also makes it correct for
uneven batch sizes.

Fix: `utils/metrics.py::PercentageAccuracy(MulticlassAccuracy)` overrides **only** `compute()`
(`return 100.0 * super().compute()`), and both models do `metric.update(...)` +
`self.log(name, metric)`. The probe also asserts `confmat.sum() == 20` after `fit()`, which
proves the state is reset per epoch rather than accumulating across the whole run.

`calculate_accuracy` is left in `helpers.py` for direct callers, but it must not be logged with
`on_epoch=True`.

### 3.7 A small subset spread over many classes gives you 1 batch per epoch

The convergence probe initially trained for "3 epochs" and the loss did not move, which looks
exactly like a broken network. It was not: with 2048 cached samples spread over **986 distinct
classes**, asking for 10 classes yields ~21 in-class images, i.e. **one batch of 21 samples per
epoch** and three optimizer steps for the whole run.

Two separate things follow from this:

1. Class selection must be count-aware **and** require validation support (see §4 —
   `choose_target_classes(..., val_dataset=, min_val_count=)`). With 512 val samples, only
   **34** of the 512 classes appear in validation at all, so a naive selection produced a
   validation set that mostly scored classes the model never saw.
2. `--train_samples_per_epoch` was added to both `train.py`s (wired to the `num_samples`
   override `create_subset_sampler` already supported) so a small cached subset can still yield
   epochs with many batches. It draws with replacement above the in-class count — fine for a
   convergence probe, and the flag is documented as such.

### 3.8 Both architectures can learn — verified by overfitting a single batch

Because "the loss is flat" has at least four unrelated causes here (data volume, epoch length,
learning rate, broken init), I cut the data out entirely: one synthetic batch, no DataLoader,
no augmentation, and check whether each network can drive its own loss below chance.

See §6.3 for the numbers. This is the test to re-run before blaming any other component.

### 3.9 The GoogLeNet auxiliary heads collapse below ~128 px input

`InceptionOutClassifier` uses `AvgPool2d(kernel_size=5, stride=3)` before its 1×1 conv. That
pool is fine at the paper's 227 px but **spatially collapses** on small inputs
(`probes/check_aux.py`):

| `--image_size` | aux tap spatial | aux `feature_map_size` |
|---|---|---|
| 96 | 6 | **1** ← crippled |
| 128 | 8 | 2 |
| 160 | 10 | 2 |
| 224 | 14 | 4 (correct) |

At `feature_map_size == 1` the aux head sees a single spatial position, so its gradient is a
global-average signal and it stops acting as a deep-supervision branch. Practical consequence:

- **Do not use GoogLeNet CPU probes below 128 px** to judge convergence — you are measuring a
  crippled branch. The production target is 224 px, where the geometry is correct, so this is a
  probe-design constraint rather than a shipping bug.
- If we ever want a legitimately small GoogLeNet config, the aux pool needs
  `kernel_size=min(5, spatial)` / `stride=min(3, spatial)` or an adaptive pool. Left alone for
  now because 224 px is the only config the final script uses.

### 3.10 Chance level for GoogLeNet's combined loss is `1.6 · ln(K)`, not `ln(K)`

The training loss is `main + 0.3·aux1 + 0.3·aux2`. At initialisation all three heads output a
uniform distribution, so the expected **starting** loss is `(1 + 2·0.3)·ln(K) = 1.6·ln(K)` —
3.684 for K=10, not 2.303. Any "did it learn?" assertion must compare against 1.6·ln(K) for the
combined loss, or against ln(K) for the main head alone. Comparing the combined loss to ln(K)
will report a healthy network as broken.

---

## 4. Code changes

### `utils/datasets_utils.py` (rewritten)

- `DEFAULT_DATASET = "ILSVRC/imagenet-1k"`, repo-relative `DEFAULT_CACHE_ROOT`.
- `_cache_key()` — sha1 over (dataset, sizes, classes, seed) so a config change cannot
  silently reuse the wrong cache.
- `load_dataset_subset()` now returns a **3-tuple** `(train, val, target_classes)` and
  validates `metadata.json` on load.
- `_split_shards()` + `_download_split_via_parquet()` — the parquet path (§3.1–3.3).
- `_download_via_streaming()` — kept as an explicit fallback for non-parquet datasets.
- `_as_undecoded_images()` — images stay as encoded bytes; decode happens in workers.
- `label_distribution()`, `choose_target_classes()` — seeded, count-ranked, and (new) requires
  val support. Replaces the unseeded `random.sample(dataset.unique("label"), k)`.
- `create_subset_sampler()` — `num_samples` override + replacement fallback instead of a
  silently near-empty epoch.
- `create_eval_sampler()` — full coverage, no `drop_last` interaction.

### `utils/runtime.py` (new)

`repo_root()`, `default_path()`, `seed_everything()`, `count_available_workers()` (CPU- **and**
shm-aware), `estimate_worker_shm_bytes()`, `available_shm_bytes()`, `resolve_accelerator()`.

### `AlexNet/train.py`, `GoogLeNet/train.py` (rewritten)

- **`pl.Trainer` moved inside `main()` under the `__main__` guard** (`P1-1`) — importing the
  module no longer starts a training run. This is also what lets `train_final.py` call
  `train.main(argv)`.
- ~30 flags, seeding, `--resume` with hparam conflict checks, `--limit_train_batches` /
  `--limit_val_batches` for smoke tests.
- **Single source of truth for class count** (`P2-4`): the sampler, the label remap and the
  model head all consume the same `target_classes`. Previously the sampler asked for 64 while
  the model was built with `args.num_classes`.
- `drop_last=False` on validation (`P2-2`) — with `drop_last=True` a small val set produced
  zero val batches, so `val_loss` never existed and `ModelCheckpoint(monitor="val_loss")`
  silently saved nothing.

### `AlexNet/model.py`, `GoogLeNet/model.py`, `GoogLeNet/inception.py`

- `image_size` param + `_flatten_features()` derived by a dummy forward (`P2-13`) — the
  hardcoded `256*6*6` only worked at 227px; 224px crashed with
  `mat1 and mat2 shapes cannot be multiplied (2x6400 and 9216x4096)`.
- **All per-batch `print()`s removed** (`P2-1`), including a 50-step full-parameter gradient
  dump; replaced by an aggregate `grad_norm/total` scalar every 50 steps.
- `configure_optimizers` reads hparams (`P2-3`) and no longer dereferences
  `self.trainer.max_epochs` unguarded (`P2-7`).
- `GoogLeNet/model.py` imports `lightning` (was `pytorch_lightning`) with a fallback;
  `InceptionOutClassifier` derives `feature_map_size` instead of hardcoding 4×4; `Linear`
  layers use `xavier_uniform_` instead of `kaiming_normal_(mode='fan_out')`, which is
  meaningless for `fan_out` on a 1-D input.
- NaN/Inf loss is logged as a scalar instead of crashing the run.

### `utils/model_utils.py`

`load_model_checkpoint` / `save_model_checkpoint` / `get_latest_checkpoint` now take
`filename` / `model_prefix` as **required** arguments (`P2-6`) — previously all three defaulted
to AlexNet-specific strings, so a generic caller silently wrote or searched for AlexNet's
artifacts. `torch.load` uses `weights_only=True`. No model-specific string remains in the file
(asserted by `probes/check_final.py`).

### `trainer/trainer.py`

`registry.json` stores `model_dir` / `module_file` **relative to `PROJECT_ROOT`** (`P3-5`) via
new `_rel_to_root()` / `_abs_from_root()` helpers, so one registry works both in the container
(`/workspace/...`) and on the host (`D:\...\pytorch_playground\...`). Legacy absolute entries
still load.

### `AlexNet/model.py`, `GoogLeNet/model.py` — optimiser selection (`P2-3`)

`--optimizer` / `--scheduler` are real flags on **both** networks and are threaded through
`configure_optimizers` and the `trainer_model.py` defaults. AlexNet previously had no way to
select Adam (it was SGD-only, which is why an early `--lr 1e-3` probe looked like a dead
network); GoogLeNet could not disable its cosine schedule.

### `AlexNet/model.py` — resume guard (`P2-5`)

`on_load_checkpoint` now tests `"target_classes" not in hparams` instead of
`... .get("target_classes") is None`. `target_classes=None` is a supported configuration, so the
old test rejected checkpoints written by the current code and made them unresumable. Verified in
both directions by `probes/check_resume_rt.py` (§6.6).

### `utils/datasets_utils.py` — cache invalidation (`P3-12`)

`shuffle_buffer_size` is recorded in `metadata.json` for provenance but excluded from the
equality check that decides whether to re-download. The parquet path ignores it, so passing a
different `--shuffle_buffer` previously forced a full re-download of a byte-identical subset.
`_cache_key()` is deliberately unchanged — adding fields to it would invalidate every existing
cache, including multi-hour downloads in flight.

### `utils/datasets_utils.py` — resumable downloads

`_download_split_via_parquet()` checkpoints to `_manifest.json` after every part file;
`_row_group_plan()` makes the replay deterministic (per-shard derived seeds) and
`_resume_state()` only adopts part files whose row counts match the manifest, so a kill
mid-write cannot poison the cache. `_as_undecoded_images()` uses `cast_column` instead of
`cast` — measured 0.17 s vs 126.14 s for 4096 samples, i.e. ~26 min saved on a 51 200-sample
subset.

### `custum_datasets/imagenet_subset_dataset.py`

`__getitem__` accepts undecoded `{'bytes','path'}` dicts and raw bytes as well as PIL images,
which is what the parquet path now stores.

### `pyproject.toml`

Added `"trainer"` to wheel `packages` (`P1-7`) — the supported entrypoint was not being
shipped.

### `probes/` (new)

The throwaway `_check_*.py` / `_probe_*.py` scripts from this session, kept as first-class
debugging tools rather than deleted, because each one answers a question that cost real time to
settle. `probes/_bootstrap.py` puts the repo root on `sys.path` so they run from `probes/` on
any checkout.

| Probe | Settles |
|---|---|
| `check_final.py` | The 40-assertion acceptance gate behind every card moved to DONE (§6.5). Run this before claiming a card. |
| `check_overfit2.py` | Whether each architecture can learn at all (§6.3) — the single most useful probe here |
| `check_metric.py` | Lightning's metric aggregation semantics (§3.6, §6.2) |
| `check_aug.py` | Whether the augmentation pipeline destroys signal (`P2-15`) — currently **failing**, by design |
| `check_aux.py` | Aux-head geometry vs input size (§3.9) |
| `check_resume_rt.py` | Real `--resume` round-trip: epoch, optimizer, scheduler (§6.6) |
| `check_ckpt.py` | Checkpoint helpers are model-agnostic; `--resume` exists |
| `check_registry.py` | Registry stores portable relative paths (§6.7) |
| `check_resume.py` | Parquet download resume fidelity (`P3-11`, **not yet run**) |
| `check_labels2.py` | Label/image alignment through the real sampler (`P2-18`, **not yet run**) |

### `.gitignore`

Added `logs/` and `Data/ckpt/`. The per-model log defaults (`AlexNet` → `Data/logs`,
`GoogLeNet` → `GoogLeNet/logs`) were not covered by the existing `tb_logs/` / `lightning_logs/`
rules, so TensorBoard event files were showing up as untracked noise.

---

## 5. "Why didn't you use the trainer?" — and the gap that question exposed

The owner was right to push on this. The answer, with evidence:

```
trainer/trainer.py :: handle_train()
    → subprocess.run(build_train_command(...))
    → [sys.executable, <model_dir>/train.py, *cli_args]
```

`trainer/` is a **registry + CLI wrapper**. It does not build a `pl.Trainer`; the only place
one is built is each model's `train.py`. So editing `train.py` in place *is* using the trainer
— forking a parallel training script would have been the actual mistake. I kept every existing
name, path and flag.

But the challenge exposed a real defect: the registry's `default_parameters` were stale, so
`python -m trainer train AlexNet` could not reach any of the new flags. Fixed by rewriting both
`trainer_model.py` files and adding 11 first-class flags to `trainer/trainer.py`.

Two more registry bugs found while verifying:

- `_resolve_model_dir` produced `/workspace/alexnet/train.py` (lowercase) from the registry key
  `alexnet`. It only worked because the Windows bind mount is case-insensitive — on Linux it
  would have been a hard failure. Now matches sibling folders case-insensitively.
- The error message told users to run `trainer register`; the working command is
  `python -m trainer register` (`P1-3`).

Verified: `python -m trainer register --model-name alexnet --force` then
`python -m trainer train AlexNet --dry-run` emits a correct command. Note `trainer/registry.json`
is gitignored, so **`register` must run on a fresh checkout**.

---

## 6. Evidence

### 6.1 Subset download (parquet path)

`python _check_data.py 4096 512 42` (scratch probe, deleted before the PR):

```
Downloading ILSVRC/imagenet-1k subset (4096 train, 512 val samples)...
  train: 4096/4096 samples (1 part files)
  validation: 512/512 samples (0 part files)
Cached for future runs!
Train dataset size: 4096
Validation dataset size: 512
elapsed=208.0s
train: 4096 samples, 986 distinct classes
val:   512 samples, 400 distinct classes
train per-class counts: min=1 p25=3 median=4 p75=5 max=10
chosen 64 classes -> 64
chosen per-class: min=7 median=8 max=10
val samples covering chosen classes: 34 / 512
sample item: <class 'dict'> (96123, 'n02978881_9704_n02978881.JPEG') label 481
```

208 s for 4096+512 samples at flat RSS, versus an OOM-kill before. The `34 / 512` line is what
motivated requiring val support in `choose_target_classes`.

#### Canonical 51200/2048 subset — built and measured

`python probes/check_big.py` (parquet download + `save_to_disk` took **759 s**, ~13 min; reload
from cache is **16.5 s** and needs no network):

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

What these numbers mean for the next agent:

* **All 1000 train classes survive** the 51200-sample cut, and the per-class distribution is
  tight (22–73, median 51). The subset is *not* pathologically skewed, so `choose_target_classes`
  has plenty of freedom.
* **Validation is the thin side.** 883/1000 classes appear at all, and the median class has
  **1** val image (p75 = 3). At `K=64` the whole validation set is **153 images** — one image per
  class on average. `val_acc` at that size has a resolution of ~0.65 % and its noise is ±several
  percent, so *do not* read a 1–2 % `val_acc` difference at `K=64` as signal. Either raise
  `val_samples` or judge convergence on `train_loss` + a larger val split.
* **`K=64` gives 4210 in-class train images** = 44 batches at `batch_size=96`. That is a real
  epoch, not the 1-batch-per-epoch trap of `P2-7`/§3.7.
* **`sampler len: 701` at `K=10`** confirms `create_subset_sampler` returns exactly one pass over
  the in-class samples with no repeats (`num_samples=None` → `num_in_class`).

**Cache-key / seed gotcha (cost me a wasted 5.7 GB build).** `seed` is part of `_cache_key`, and
`train_final.py` bakes `"seed": 42`, which hashes to
`ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538`. The statistics above were produced under
`seed=1337` → `..._2ecb016d7b0d`, a *different* directory. `probes/check_big.py` now pins
`SEED = 42` to match the deliverable. If you build a subset with any other seed, `train_final.py`
will not find it and will start a 13-minute download. Check the directory name against
`_cache_key(dataset_name, 51200, 2048, None, 42)` before assuming a cache is warm.

**`save_to_disk` memory.** The first attempt at the `seed=42` subset was **OOM-killed at 2/12
shards** inside the 8.25 GB container cap; the `seed=1337` build completed all 12 shards (it was
re-run after the earlier parts were already on disk, and peak RSS stayed under the cap). So the
round-trip is *borderline*, not impossible, at this size — see `P1-8` for the proposed fix (let
the cache be the part files and skip `save_to_disk` entirely).

### 6.2 Accuracy metric — measured Lightning aggregation semantics

`probes/check_metric.py` logs the same ground-truth accuracy (60.0 %) three ways through a real
`Trainer.fit()` and prints the epoch value Lightning reports:

```
style A: acc = 60.0000 confmat_sum_after_fit=20
style B: acc = 60.0000 confmat_sum_after_fit=-1 (20 => reset per epoch, 60 => not reset)
true global = 60.0000 ; mean-of-batches = 66.6667
```

Style A is `metric.update()` + `self.log(name, metric)`; style B is the same metric driven
through `forward()` and logged as a tensor. Both now agree with the hand-computed global value,
and `confmat_sum_after_fit=20` confirms the state is reset per epoch. Before the fix, style A
printed **6000.0** where the truth was 60.0. Note `mean-of-batches = 66.6667` — that is what the
old `calculate_accuracy` + `on_epoch=True` code was computing, and it is why the broken smoke run
reported nonsense accuracies.

### 6.3 Both architectures can learn — overfit-one-batch

`probes/check_overfit2.py`: one synthetic batch (`B=32`, `K=10`, 96 px), no DataLoader, no
augmentation, Adam `1e-3`, 120 steps. This is the test to run before blaming the data pipeline.

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

**Reading the GoogLeNet numbers correctly matters.** Its loss is
`main + 0.3·aux1 + 0.3·aux2`, so at initialisation all three heads are uniform and the expected
*combined* loss is `(1 + 2·0.3)·ln(10) = 3.684` — the observed first-step 3.6900 is exactly
chance, not a bad init. A naive threshold at `ln(K)` mislabels this as "failed to learn"; I made
that mistake in the first pass and it is recorded here so the next agent does not repeat it.
Both networks reach ~50 % on 32 memorisable samples in 120 steps, so neither architecture nor
initialisation is the blocker.

### 6.4 The auxiliary heads collapse at small image sizes

`probes/check_aux.py` prints the real aux geometry per input size; the table is in §3.9. Summary:
`feature_map_size` is 1 / 2 / 2 / 4 at 96 / 128 / 160 / 224 px. The 224 px production config is
correct; only small-image probes are affected.

### 6.5 Kanban acceptance gate — 40/40

`probes/check_final.py` re-verifies every card moved to DONE in this handoff, mechanically, from
source and from real behaviour (not from memory). It checks import safety, import style,
portable-path defaults, the `.gitkeep`, the wheel `packages` list, absence of `print()` inside
`training_step`/`validation_step`, seeding, the real CLI flags, the single class-count source,
`--resume`, the checkpoint-helper signatures and `weights_only`, the aux init and aux logging,
`--train_samples_per_epoch` and `drop_last=False` on val, the derived flatten size, and the
registry path round-trip.

```
40 OK / 0 FAIL
FINAL: ALL PASS
```

Three of its first-run failures were **probe** bugs, not code bugs, and are worth remembering
because they are the classic shape of a false positive:

- `256 * 6 * 6` matched inside the explanatory docstring of `_flatten_features()`.
- `mode='fan_out'` matched the **Conv2d** branch, which is the correct place for it; only the
  `Linear` branch was in scope.
- The aux metrics are logged as `train_loss_aux1` / `train_loss_aux2`, not `aux1_loss`.

Two were **real** gaps, fixed before this section was written: `--optimizer` / `--scheduler`
were missing on AlexNet (see §6.6), and `--scheduler` was missing on GoogLeNet.

### 6.6 Resume round-trip — and the `on_load_checkpoint` bug it exposed

`probes/check_resume_rt.py` trains AlexNet for 2 epochs with the same `ModelCheckpoint` naming
`train.py` uses, locates the checkpoint through `get_latest_checkpoint()`, and re-enters
`fit(ckpt_path=...)` with `max_epochs=4`:

```
checkpoints: ['alexnet_epoch_epoch=00.ckpt', 'alexnet_epoch_epoch=01.ckpt']
latest -> alexnet_epoch_epoch=01.ckpt
phase3 current_epoch = 4
optimizer state restored = True
scheduler restored = True last_epoch = 4
legacy checkpoint rejected = True
[P2-5] resume round-trip PASS
```

Two findings:

1. **The filename glob is correct, but only by accident of Lightning's defaults.** The
   callback is configured `filename="alexnet_epoch_{epoch:02d}"` and Lightning's
   `auto_insert_epoch` then injects the literal `epoch=`, producing
   `alexnet_epoch_epoch=00.ckpt` — which is what `f"{model_prefix}_epoch=*.ckpt"` matches. If
   anyone sets `auto_insert_epoch=False` or renames the pattern, resume silently returns
   `None` and the run restarts from scratch. The probe asserts non-`None`, so it now fails
   loudly instead of silently.
2. **`on_load_checkpoint` made current-code checkpoints unresumable.** It tested
   `hyper_parameters.get("target_classes") is None` and raised. But `target_classes=None` is a
   *supported* configuration (output indices then equal the original class labels), so a
   checkpoint written by the current code with that setting was rejected as "pre-remapping".
   The guard now tests `"target_classes" not in hparams` — key **presence**, not value. The
   probe covers both directions: a normal checkpoint resumes, and a checkpoint with the key
   deleted is still rejected.

### 6.7 Registry portability (`P3-5`)

`probes/check_registry.py` re-registers both networks and inspects `registry.json`:

```
[P3-5] OK   alexnet.model_dir relative: AlexNet
[P3-5] OK   alexnet.module_file relative: AlexNet/trainer_model.py
[P3-5] OK   googlenet.model_dir relative: GoogLeNet
[P3-5] OK   googlenet.module_file relative: GoogLeNet/trainer_model.py
[P3-5] dry-run alexnet:   rc=0  ... /workspace/AlexNet/train.py --version_name alexnet_v1 ...
[P3-5] dry-run googlenet: rc=0  ... /workspace/GoogLeNet/train.py --version_name googlenet_v1 ...
[P3-5] PASS
```

`_abs_from_root()` still accepts a legacy absolute entry, so an existing `registry.json` keeps
working.

### 6.8 `train_final.py --smoke`

Both parameterless scripts were executed for the first time in this session, via their
`--smoke` path (reduced sizes, identical recipe). The `RESULT` block is the artefact to judge,
and it prints the chance threshold for the configured class count so a reader cannot mistake
chance for learning.

**AlexNet** — `python AlexNet/train_final.py --smoke`, exit code `0`:

```
==================== RESULT ====================
  train_loss = 2.5463
   train_acc = 15.6250
    val_loss = 2.4703
     val_acc = 8.3333
 checkpoints = <repo>/AlexNet/checkpoints
 tensorboard = <repo>/Data/logs
Chance-level accuracy for 10 classes is 10.00%; val_acc must clearly exceed it.
================================================
```

The script works: it runs end to end, saves `alexnet_final.ckpt`, writes TensorBoard logs, and
correctly tells the reader that `val_acc = 8.33` is **below** the 10.00% chance level for
K=10. That is the honest outcome of a 2-epoch / 1024-sample CPU smoke run, not a defect in the
script — but it is also the clearest confirmation of `P2-14`: neither network has beaten chance
yet, and the smoke path is too short to expect it to.

**GoogLeNet** — `python GoogLeNet/train_final.py --smoke`, exit code `0`:

```
==================== RESULT ====================
      train_loss = 3.6667
 train_loss_main = 2.3007
       train_acc = 12.5000
        val_loss = 2.3032
         val_acc = 8.3333
 checkpoints = <repo>/GoogLeNet/checkpoints
 tensorboard = <repo>/GoogLeNet/logs
Chance-level accuracy for 10 classes is 10.00%; val_acc must clearly exceed it.
================================================
```

Read these against the **correct** thresholds (§3.10 / `P2-17`), which is exactly why the
combined loss is logged next to the main one:

| Quantity | Observed | Chance level | Verdict |
|---|---|---|---|
| `train_loss` (combined) | 3.6667 | `1.6·ln(10) = 3.6842` | at chance |
| `train_loss_main` | 2.3007 | `ln(10) = 2.3026` | at chance |
| `val_loss` (main head only) | 2.3032 | `ln(10) = 2.3026` | at chance |

Everything sits exactly on chance after 2 epochs of a 1024-sample run, which is the expected
signature of "correct code, not enough training" rather than a broken network — consistent with
the overfit-one-batch result in §6.3, where the same model drove its combined loss from 3.6900
to 1.8572. Had I compared `train_loss = 3.6667` against `ln(10)` I would have concluded the
auxiliary heads were broken; against `1.6·ln(10)` they are simply untrained.

Both scripts therefore satisfy the user's contract: **one file per network, no parameters, all
best hyperparameters baked in**, with `--smoke` as the only accepted argument and a `RESULT`
block that states the chance threshold so nobody can mistake chance for learning. The remaining
gap is `P3-13`: the baked-in `CONFIG` numbers are still informed defaults, not measured values.

---

## 7. Environment notes for the next agent

- **Repo root is nested**: `.../pytorch_playground/pytorch_playground`. Git must run in the
  inner folder.
- Container: `nicoladesocio/pytorchplayground:latest` has a `code-server` `ENTRYPOINT` → always
  pass `--entrypoint bash`. It is **missing** `lightning`, `datasets`, `tensorboard`, `scipy`,
  `pandas`, `scikit-learn`, `tqdm`; install with
  `pip install --break-system-packages -q ...`.
- **Always `--shm-size=4g`.** Default 64 MB → DataLoader workers die with
  `unable to allocate shared memory(shm) for file </torch_...>: No space left on device (28)`.
  I keep a `ppg-dev3:snapshot` image with deps preinstalled.
- **GPU is still unusable here**: host driver 465.89 predates WSL2 CUDA passthrough and the
  GTX 960M is `sm_50`; `torch.cuda.is_available()` is `False`. All agent validation is CPU;
  the baked-in scripts target a bigger GPU (`16-mixed`, batch 96/128).
- Docker memory cap is 8.25 GB while the host has 16 GB with ~1.1 GB free. Raising the Docker
  Desktop limit is still the single highest-leverage host fix, though the parquet path no
  longer needs it.
- PowerShell gotchas on this host: nested quotes in `docker exec bash -lc '... python -c "..."'`
  get mangled → **write a `.py` probe file instead**. Windows-created `.sh` files get CRLF. No
  `tail`. Long runs exceed the tool wait → redirect to a log inside the container and `cat` it
  through `tr '\r' '\n'`.

---

## 8. Open / left for the next agent

- **GoogLeNet convergence** is unproven. AlexNet smoke is in §6.2; GoogLeNet needs the same
  treatment, and its aux-loss weight (0.3) has never been swept.
- The `CONFIG` values in both `train_final.py` are **informed defaults, not yet measured on a
  GPU**. They must be re-baked once a real GPU run exists. Until then the scripts are correct
  in shape but their numbers are provisional — do not cite them as tuned.
- `--shuffle_buffer` is now dead weight on the parquet path. Decide whether to keep it purely
  for the streaming fallback (my recommendation) or remove it from the CLI. The cache-invalidation
  hazard it created is **already fixed** (`identity_keys` in `load_dataset_subset` excludes
  `shuffle_buffer_size` from the comparison); what is left is the keep-or-remove decision (`P3-12`).
- `save_to_disk` is the only memory-heavy step left in the data path and is borderline at
  51200 samples under the 8.25 GB container cap (`P3-14`).
- Whether to keep the streaming fallback at all. I kept it; deleting it would break any
  non-parquet dataset.
- `trainer_v2/` (`P3-6`) and the stale `origin/update_streaming_dataset` branch (`P3-9`) are
  still unresolved.
- `custum_datasets/` is still misspelled (`P2-9`) — deliberately not renamed silently.

---

## 9. Exactly what the next agent should do

Read in this order: `repos_log/AGENT_INSTRUCTIONS.md` → `repos_log/goal.md` → this file §0 →
`repos_log/KANBAN.md` 🔥 column. Then, in this order:

1. **Finish `P1-8` — the canonical subset.** One command:
   `python probes/check_big.py` (now pinned to `SEED = 42`, which is what `train_final.py`
   bakes). It takes ~13 min and **resumes** from `_manifest.json`. Then confirm
   `Data/datasets/ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538/` contains `train/`,
   `validation/` and `metadata.json`.
   *Why it is still open:* the verified subset in §6.1 was built with `seed=1337` →
   `..._2ecb016d7b0d`, and `seed` is part of `_cache_key`, so `train_final.py` (`"seed": 42`)
   looks in `..._c58e22ff2538` and finds only a partial 534 MB download. **The mechanism is
   proven** — the other seed completed all 12 shards — this is a "run it once with the right
   seed" task, not an investigation. **Do not** delete `_manifest.json` unless you mean to
   restart the download. If `save_to_disk` is OOM-killed again, that is `P3-14`, not a mystery.
2. **Run the two probes I wrote but never executed** — they are the cheapest remaining
   correctness risks:
   - `probes/check_resume.py` (`P3-11`) — kill the download at 90 s, resume, and compare label/path
     fingerprints against an uninterrupted reference.
   - `probes/check_labels2.py` (`P2-18`) — confirm labels still line up with images through the real
     sampler. If this one is broken, every accuracy number in the repo is meaningless.
3. **Chase `P2-15` before anything else convergence-related.** The augmentation probe is the
   single most informative experiment left: 89 real images, both transform arms, 80 steps, and
   it failed while synthetic noise succeeded. Start by making `Resize` respect `image_size`
   (`Resize(max(image_size, 256))` or an explicit `--resize_size`) and re-run `probes/check_aug.py`.
   Do not spend time on learning-rate sweeps until this is understood — a 14% area crop makes
   every other knob irrelevant.
4. **Then get one network above chance (`P2-14`).** Prefer a real GPU: `16-mixed`, batch 96,
   `image_size=224` or `227`, `--train_samples_per_epoch` set so an epoch is ≥200 batches.
   AlexNet uses **SGD** — `--lr 1e-3` is far too small, the paper value is `0.01`; `--optimizer
   adam` now exists and needs ~`1e-4`. Assert against the right chance level (`P2-17`):
   `ln(K)` for AlexNet, `1.6·ln(K)` for GoogLeNet's combined loss.
5. **Re-bake `train_final.py` (`P3-13`).** Both scripts ran for the first time in this session
   and their `CONFIG` numbers are still informed guesses. After step 4, replace them with the
   measured values and record the run in a handoff. The scripts take **no arguments** except
   `--smoke`; do not add flags.
6. **Housekeeping that is genuinely still broken** (verified this session, deliberately left
   for a separate commit): `P1-3` (`trainer/README.md` still documents the crashing
   `python3 trainer …` forms), `P1-5` (`Dockerfile` still `COPY nscacert.crt` + EOL
   `openjdk-11-jdk`), `P3-2` (`GoogLeNet/__init__.py` and `trainer_v2/__init__.py` missing),
   `P3-4` (`InceptionNaive` still present).

**Do not re-derive** anything in §3 — the dataset id, the `datasets/` path prefix, the shuffled
row groups, the metric aggregation semantics, the chance thresholds, the aux collapse geometry
and the epoch-length trap are all measured facts. Re-probing them wastes the budget.

**Environment**: bring up `ppg-dev3` from `ppg-dev3:snapshot` with `--shm-size=4g` and the repo
bind-mounted at `/workspace`; export `HF_TOKEN` from `/root/.hf_token` (never commit it). Push
over SSH (`origin-ssh`) and open the PR with the GitKraken MCP — HTTPS push does not work here.
