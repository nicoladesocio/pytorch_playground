# Shared Kanban — pytorch_playground

> Board for all agents working on this repo. Read [goal.md](./goal.md) first.
>
> **How to use**
> - Move a card between columns in the same commit that does the work.
> - Cards are stable IDs — reference them in commits, PRs, and handoffs (`P1-3`).
> - Add a `Owner` and `Handoff` when you pick a card up. Add new cards at the bottom of
>   `BACKLOG` with a `P1/P2/P3` priority; don't renumber existing cards.
> - A card moves to `DONE` only when its acceptance criteria are met **and** evidence
>   (log/metric excerpt) is in a handoff.
>
> Legend — **P1** blocks any training run · **P2** correctness/reproducibility · **P3** polish

---

## 🔥 IN PROGRESS

| ID | Card | Owner | Handoff |
|---|---|---|---|
| `P2-11` | AlexNet convergence — overfit-one-batch **passes** (2.554 → 1.189), but 18 epochs of SGD `lr=0.01` on the 4096-sample subset left `val_acc` pinned at 8.330 (chance for K=10). Needs a longer/warmer schedule or a GPU run, and the augmentation question below resolved. | agent/data-unblock-and-convergence | [003](./handoffs/HANDOFF_003.md) |
| `P1-8` | Canonical 51200/2048 subset must finish downloading and be verified. The resumable parquet path is new; the first run exposed and fixed a real bug (`_resume_state()` arity). | agent/data-unblock-and-convergence | [003](./handoffs/HANDOFF_003.md) |
| `P2-15` | `ImageNetSubsetDataset` hardcodes `Resize(256)` before `RandomCrop(image_size)`. Both the train and eval transform arms **failed to memorise 89 real images in 80 steps** while synthetic noise memorised fine — the top open convergence suspect. | agent/data-unblock-and-convergence | [003](./handoffs/HANDOFF_003.md) |

---

## 👀 REVIEW / WAITING

| ID | Card | Blocked on | Owner |
|---|---|---|---|
| — | *nothing waiting* | | |

> `P2-11` (AlexNet convergence) moved to 🔥 IN PROGRESS in HANDOFF_003 — every blocker
> listed for it (`P1-1`, `P1-2`, `P1-4`, `P2-1`) is now DONE.

> `trainer_v2/`'s fate is tracked once, as `P3-6` in the P3 list below (blocked on a product
> decision from the repo owner). It was previously duplicated here.

---

## ✅ DONE

| ID | Card | Done in | Handoff |
|---|---|---|---|
| `P0-1` | Explore repo; audit good / bad / missing | HANDOFF_001 | [001](./handoffs/HANDOFF_001.md) |
| `P0-2` | Create `repos_log/` with curated `goal.md` + `handoffs/` | HANDOFF_001 | [001](./handoffs/HANDOFF_001.md) |
| `P0-3` | Create this shared kanban | HANDOFF_001 | [001](./handoffs/HANDOFF_001.md) |
| `P0-4` | Write `AGENT_INSTRUCTIONS.md` — reading order, PR-as-deliverable contract, Docker/CPU runbook | HANDOFF_002 | [002](./handoffs/HANDOFF_002.md) |
| `P0-5` | Verify the Docker environment: image runs, deps installable, **both models complete `Trainer.fit()` on CPU**, GPU confirmed unusable | HANDOFF_002 | [002](./handoffs/HANDOFF_002.md) |
| `P1-1` | `AlexNet/train.py` `Trainer` block moved inside `__main__`; module is now import-safe (`build_arg_parser()` / `main(argv)` split) | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P1-2` | Lightning imports standardised to `import lightning as pl`; `GoogLeNet` intra-package imports made package-qualified so `import GoogLeNet.model` works from the repo root | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P1-4` | Portable paths via `utils/runtime.py` (`repo_root()` / `default_path()`); no `/workspace` defaults remain; honours `PYTORCH_PLAYGROUND_ROOT` | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P1-6` | `Data/datasets/.gitkeep` committed (`.gitignore` negation verified to admit it) | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P1-7` | `trainer` added to `pyproject.toml` wheel `packages` | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-1` | Per-batch `print()` removed from train/val steps; NaN/range violations became logged metrics | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-2` | Seeding: `seed_everything(seed)` in both train scripts, seeded class selection, `--seed` flag | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-3` | `--max_epochs`, `--lr`, `--optimizer`, `--scheduler` are real CLI args; `trainer train <M> --set max_epochs=5` works | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-4` | Single source of truth for class count — sampler, dataset remap and model head all consume the same `target_classes` | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-5` | Explicit `--resume` on both nets; fresh runs unaffected. **Extra bug found + fixed while verifying:** `on_load_checkpoint` rejected any checkpoint whose `target_classes` was `None`, which made a checkpoint written by the *current* code unresumable. Now only a genuinely pre-remapping checkpoint (key absent) is rejected. Round-trip verified: epoch, optimizer momentum and scheduler all restored | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-6` | `utils/model_utils.py` is model-agnostic: `filename`/`model_prefix` are required args, no AlexNet strings remain, `torch.load(weights_only=True)` | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-7` | Aux weight configurable; `InceptionOutClassifier` Linear init uses `fan_in`; aux and main losses logged separately | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-10` | Epoch size explicit and logged (`--train_samples_per_epoch`); val loader never `drop_last`; warns when in-class count < `batch_size` | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P2-13` | AlexNet flatten size derived from its own conv stack (`_flatten_features()`); bad size raises a clear message | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |
| `P3-5` | `registry.json` stores paths **relative to `PROJECT_ROOT`**; legacy absolute entries still readable via `_abs_from_root()`. Verified: re-registering both nets yields relative paths and `trainer train --dry-run` resolves them in-container | HANDOFF_003 | [003](./handoffs/HANDOFF_003.md) |

> Evidence for the `P1-*` / `P2-*` / `P3-5` rows above: `HANDOFF_003` §6 plus the
> acceptance probes (`12/12` kanban-claim checks, checkpoint-helper checks, resume
> round-trip). The data-path unblock itself — `ILSVRC/imagenet-1k` as the correct Hub id,
> the parquet row-group downloader, and the `PercentageAccuracy` fix — is the substance
> of HANDOFF_003.

---

## 📋 TODO — P1 (blocks every training run)

| ID | Card | Acceptance criteria | Est. |
|---|---|---|---|
| `P1-1` | **Fix `AlexNet/train.py`: `Trainer` block escapes `__main__` guard.** Lines 73–93 (`trainer = pl.Trainer(...)` … `save_model_checkpoint(...)`) are module-level, so they execute on *import* — including `import model`-adjacent imports and any test collection. `lightning_model`/`args` are only bound inside the guard. | `python -c "import AlexNet.train"` is a no-op; whole body indented under `__main__`; smoke train completes. | S |
| `P1-2` | **Standardise Lightning imports + fix intra-package imports.** ~~Declare `pytorch_lightning`~~ — **claim corrected in HANDOFF_002**: `lightning` 2.6.6 *does* depend on `pytorch_lightning`, so both import fine. The real problems are (a) two inconsistent import styles across models, and (b) `GoogLeNet/model.py:4` uses `from inception import ...` and `GoogLeNet/train.py:4` uses `import model` — bare top-level imports that only resolve if `GoogLeNet/` is itself on `sys.path`, so `import GoogLeNet.model` fails from the repo root. | All modules use `import lightning as pl`; intra-package imports are relative or package-qualified; `python -c "import GoogLeNet.model, GoogLeNet.train"` succeeds from the repo root. | S |
| `P1-3` | **Fix `trainer` CLI invocation.** `python trainer …` and `python trainer/trainer.py …` (both documented in `README.md` / `trainer/README.md`) crash with `ModuleNotFoundError: No module named 'trainer.model_base'; 'trainer' is not a package` — the script's own folder lands on `sys.path[0]` and shadows the `trainer` package. Only `python -m trainer` works. | All three documented forms work, or docs + `trainer/README.md` state `python -m trainer` as the only supported form and the script-form fallback is removed. Add a smoke test. | S |
| `P1-4` | **Portable paths.** Every default is container-only (`/workspace/AlexNet/checkpoints`, `/workspace/Data/datasets/`, `/workspace/GoogLeNet/logs`). Nothing runs outside the devcontainer. | Paths derive from `PROJECT_ROOT` (or a `--data-root` / `PYTORCH_PLAYGROUND_ROOT` env var); defaults resolve correctly on Linux, in-container, and on Windows. | M |
| `P1-5` | **`Dockerfile` build is broken.** `COPY nscacert.crt …` references a gitignored, absent file → `docker build` fails at step 1. Also pins `openjdk-11-jdk` (EOL) purely for a corporate cert. | Image builds from a clean checkout; corporate-cert handling is opt-in (build arg / separate stage), not mandatory. | M |
| `P1-6` | **`Data/datasets/` placeholder missing.** `.gitignore` whitelists `!Data/datasets/.gitkeep` but the directory doesn't exist, so the documented cache root has no anchor and `load_dataset_subset` must `makedirs` blind. | `Data/datasets/.gitkeep` committed; cache root exists after clone. | XS |
| `P1-7` | **`pip install -e .` does not install the `trainer` CLI.** `pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages` lists `AlexNet`, `GoogLeNet`, `utils`, `custum_datasets`, `trainer_v2` — but **not `trainer/`**. Verified by building the wheel in HANDOFF_002: `trainer/` is absent from the archive. So the one working entrypoint (`python -m trainer`) only works from a source checkout, never from an installed package. | `trainer` in `packages`; built wheel contains `trainer/*.py`; `python -m trainer --help` works in a venv with only the wheel installed. | XS |

---

## 📋 TODO — P2 (correctness, reproducibility, observability)

| ID | Card | Acceptance criteria | Est. |
|---|---|---|---|
| `P2-1` | **Remove per-batch `print()` from training/validation steps.** `AlexNet/model.py` prints per-batch loss/acc plus a `Grad <name>` dump over *every* parameter for the first 50 steps (~1.5k lines once), and prints an "Invalid data" line whenever a NaN/range check trips; `GoogLeNet/model.py` prints per-batch losses in both train and val. These bypass the logger, interleave badly with the progress bar, and make logs unparseable. | Metrics flow through `self.log(...)` / TensorBoard only; no `print` inside `training_step`/`validation_step`; NaN/range violations become logged metrics or raised errors instead of stdout. | S |
| `P2-2` | **Add seeding.** Zero `seed_everything` / `manual_seed` anywhere in the repo. `create_subset_sampler` uses unseeded `random.sample`, so the 64-class subset differs every run — runs are not comparable and resumed runs can silently drift. | `pl.seed_everything(seed)` in both train scripts; class-subset selection seeded and persisted; `--seed` CLI flag; same seed ⇒ same metrics. | M |
| `P2-3` | **Expose `max_epochs` (and optimizer/scheduler) as parameters.** `max_epochs` is hardcoded (AlexNet 100, GoogLeNet 60). The documented `--set max_epochs=20` therefore emits `--max_epochs 20`, which `argparse` rejects with an "unrecognized arguments" error. | `--max_epochs`, `--lr`, `--optimizer`, `--scheduler` are real CLI args; `trainer train <M> --set max_epochs=5` works. | M |
| `P2-4` | **Fix AlexNet `num_classes` / `target_classes` mismatch.** `--num_classes` (default 64) and `create_subset_sampler(max_num_classes=64)` are independent knobs; AlexNet hardcodes `64` for the sampler while passing `args.num_classes` to the model. Any other value ⇒ out-of-range labels vs. classifier head. GoogLeNet correctly threads `args.num_classes` into both. | Single source of truth for class count; mismatch raises a clear error before training. | S |
| `P2-5` | **AlexNet resume path is broken.** `train.py` calls `get_latest_checkpoint(args.checkpoint_path)` unconditionally (no `--resume` flag, unlike GoogLeNet) and then `trainer.fit(..., ckpt_path=checkpoint_path)`. Combined with `on_load_checkpoint` raising `ValueError` on checkpoints lacking `target_classes`, an interrupted run cannot be restarted. | Explicit `--resume`; fresh runs unaffected; resume restores optimizer/scheduler/epoch; test covering both paths. | M |
| `P2-6` | **`save_model_checkpoint` hardcodes `alexnet_final.ckpt`** in `utils/model_utils.py` while being generic (`model`, `checkpoint_root`). If reused it silently overwrites AlexNet weights. Also `torch.load` without `weights_only=True`. | Filename derived from model/version; `weights_only=True`; no AlexNet-specific strings in generic helpers. | S |
| `P2-7` | **`GoogLeNet` aux-loss weight + `InceptionOutClassifier` init.** Aux weight is hardcoded `0.3` (paper: 0.3, but should decay to ~0 late in training). `InceptionOutClassifier` applies `kaiming_normal_(mode='fan_out')` to the **Linear** layers too — `fan_out` is the wrong mode for the classifier and `mode='fan_in'` is intended for forward-pass ReLU layers. | Aux weight configurable + annealable; Linear init uses `fan_in`; aux-vs-main loss curves logged separately and both decreasing. | M |
| `P2-8` | **`AlexNet/test_synthetic.py` is not a test.** It's a script that unconditionally calls `.cuda()` (crashes on CPU-only hosts), allocates `100×3×227×227` fp32 (~600 MB) up front, and asserts nothing — so it can't be run by pytest or CI. | Converted to a real pytest test with `@pytest.mark.skipif` on no-CUDA (or CPU-forced), tiny shapes, and an assertion that loss decreases. | M |
| `P2-9` | **No test suite, linter, formatter, or CI.** `pyproject.toml` has no `[project.optional-dependencies]`, no pytest/ruff/black config, no CI workflow. Nothing catches the `__main__`-guard or import bugs above. | `pytest`, `ruff`, `black` configured + dev extras; GitHub Actions workflow runs lint + CPU smoke tests on PR. | M |
| `P2-10` | **`create_subset_sampler` silently shrinks epochs / can starve validation.** `num_samples=num_in_class` with `replacement=False` means an epoch only covers the in-class fraction of the cached subset — with 51 200 cached samples and 64/1000 classes that's ~3 300 real images per epoch, so `--train-samples` is misleading. `val_samples=2048` likewise yields ~130 usable val images, and `drop_last=True` can drop the *entire* val loader if <1 batch survives. | Epoch size is explicit and logged; validation never silently empties; clear error/warning when in-class count < `batch_size`. | M |
| `P2-12` | **The Dockerfile in the repo does not build the image actually in use.** `nicoladesocio/pytorchplayground:latest` (33.5 GB, present locally) was built from a **different** Dockerfile: it installs `code-server` 4.96.4 with an `ENTRYPOINT`, and ships `torch 2.13.0+cu132` / Python 3.12, whereas the repo `Dockerfile` pins `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` and has no `code-server`. Consequences: (a) `docker run <image> python3 …` fails with `exec: "code-server": executable file not found in $PATH`; (b) the image is **missing** `lightning`, `datasets`, `tensorboard`, `scipy`, `pandas`, `scikit-learn`, `tqdm`. Note `pip install .` of a hatchling project whose `packages` dirs are absent still **succeeds silently**, so the dependency layer can pass while installing nothing. | Repo `Dockerfile` builds reproducibly and the resulting image imports every declared dep; `docker run --rm <image> python -c "import lightning, datasets"` passes; no `code-server` ENTRYPOINT surprise (or it is an explicit opt-in stage). | M |
| `P2-13` | **AlexNet classifier hardcodes `256*6*6` — latent shape bug.** `AlexNet/model.py:39` uses `nn.Linear(256 * 6 * 6, 4096)`, but conv1 is `stride=4` with **no padding**, so spatial size is `floor((H-11)/4)+1` then pooled. Verified in HANDOFF_002: `224→6400` (**RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x6400 and 9216x4096)**), `227/231/256→9216` OK. It works today *only* because both `train.py` files happen to pass `image_size=227`, while `ImageNetSubsetDataset`'s own default is `image_size=224`. Any caller using the default hits the crash. | Flatten size derived from the model's own conv stack (or an explicit `input_size` arg validated at construction); incompatible size raises a clear message naming the expected size; a test covers 224 and 227. | S |

---

## 📋 TODO — P3 (polish, hygiene, docs)

| ID | Card | Acceptance criteria | Est. |
|---|---|---|---|
| `P3-1` | Rename `custum_datasets/` → `custom_datasets/` (typo, referenced in ~6 files + `pyproject.toml`). | Rename via IDE/refactor tool; all imports + packaging updated; no `custum` left in repo. | S |
| `P3-2` | Add missing `__init__.py` to `GoogLeNet/` and `trainer_v2/`; both are listed as wheel packages in `pyproject.toml` yet are namespace packages, and `GoogLeNet/train.py` relies on `import model` (implicit relative) which only works when CWD is `GoogLeNet/`. | Both are real packages; imports are absolute; `pip install -e .` then import from any CWD. | S |
| `P3-3` | `AlexNet/run.py` inference script is broken: it builds an **untrained** `AlexNet` and never loads a checkpoint (`load_model_checkpoint` exists but is unused), shadows the `model` module with a local variable named `model`, and defaults `--num_classes 10` vs. training's 64. | Loads weights from a checkpoint; predicts real classes; maps indices back via `predict_classes`. | M |
| `P3-4` | Dead / duplicated code: `InceptionNaive` (unused, non-paper variant); commented-out optimizers, softmaxes, and debug prints across `GoogLeNet/model.py` + `inception.py`; `AlexNet/model.py` `debug_features()` unused; `AlexNet/trainer_model.py` `name = "alexnet"` vs. registry key `AlexNet` inconsistency. | Dead code removed or documented; no commented-out code blocks. | S |
| `P3-5` | `trainer/` registry stores **absolute** `model_dir`/`module_file` paths in `registry.json`, so a registry generated in the container (`/workspace/...`) is invalid on the host and vice versa. Also `registry.json` is gitignored, so every fresh checkout must re-register. | Paths stored relative to `PROJECT_ROOT`; consider committing a default registry or auto-discovering model folders. | S |
| `P3-6` | `trainer_v2/` is a non-functional draft: `trainer()` calls `load_module("trainer_v2.trainer", "default", ...)` which instantiates the **module-level function** `default` — but `DATASET_CACHE_ROOT` points at `/workspace/Data/checkpoints/` while `trainer()`'s own checkpoints go to the same dir; `dataset()` returns a bare `(train, val)` HF tuple that `Trainer.fit` cannot consume; `model()` passes `target_classes=None` into modules that expect it; `pyproject.toml` omits `trainer/` from `packages`. | Either a working vertical slice with tests, or move to `docs/designs/` and delete the code. Decision recorded in a handoff. | L |
| `P3-7` | `AlexNet` uses `image_size=227` with `Resize(256)` + `RandomCrop(227)`; the paper resizes to 256 and crops 224, and `GoogLeNet` also uses 227. Inconsistent + off-by-one vs. paper. `RandomCrop` without padding also drops aspect-ratio handling. | Documented, consistent input pipeline across models; comment explains any deliberate deviation. | S |
| `P3-8` | Docs drift: root `README.md` "Typical Workflow" uses the broken `python3 trainer …` form; `trainer/README.md` "Next Steps" duplicates it; `AlexNet/Status.md` says "kaiman" (→ Kaiming) and lists tried fixes without results. | Docs match working commands; typos fixed; Status.md updated with actual outcomes. | S |
| `P3-9` | Review the unmerged remote branch `origin/update_streaming_dataset` (2 commits: `StreamingDataset` + arrow memory-map/prefetch, +508 lines, new `local_image_cache_dataset.py` / `streaming_imagenet_dataset.py`). It likely supersedes `utils/datasets_utils.py`'s blocking download-then-train flow. | Branch reviewed; either merged via PR, superseded by a better approach, or explicitly closed. Decision in a handoff. | M |
| `P3-10` | Add a `README.md` in `repos_log/` explaining the folder contract (goal vs. kanban vs. handoffs) so new agents don't have to infer it. | File exists and matches `goal.md` ground rules. | XS |

---

## 🧊 BACKLOG / IDEAS (unscheduled)

- **Streaming dataset**: adopt `origin/update_streaming_dataset`'s overlap-download-and-train design so training starts immediately (`P3-9`).
- **Precision**: both train scripts pin `precision="32"`. Try `16-mixed` / `bf16-mixed` — likely essential on a 4 GB GPU.
- **Overfit-one-batch sanity check** as a first-class debugging tool (the fastest way to settle the AlexNet gradient question).
- **Metrics persistence**: `metrics.json` per run (design exists in `trainer_v2/README.md`) so results are diffable without TensorBoard.
- **Experiment registry**: record run config + result per run to compare AlexNet/GoogLeNet attempts over time.
- **`ImagesFolderDataset`**: `os.listdir` is not recursive and it returns `(image, filename)` — unusable for classed inference; consider `ImageFolder` from torchvision.
- **`visualize_weights.py`**: AlexNet and GoogLeNet copies duplicate ~80 % of each other; unify into `utils/`.
- **GPU strategy for this host**: ✅ resolved in HANDOFF_002 — CPU-only is the supported
  experiment path; see [AGENT_INSTRUCTIONS.md §4](./AGENT_INSTRUCTIONS.md).

### Added by HANDOFF_003

| ID | Priority | Card | Acceptance criteria |
|---|---|---|---|
| `P1-8` | **P1** | Canonical 51200/2048 subset must finish downloading and be verified. The resumable parquet path is new; its first run exposed a real bug (`_resume_state()` was called with 4 args but defined 3). | Subset exists under the cache root with `metadata.json`; distinct-class count and per-class percentiles logged; `K=10/64/128` in-class train **and** val counts printed; load from cache is offline-fast. |
| `P2-14` | **P2** | Neither network has ever beaten chance. AlexNet: 18 epochs, SGD `lr=0.01`, 4096-sample subset → `val_acc` pinned at 8.330 with `train_loss_epoch` 2.660 → 2.430. Needs a GPU run, a longer/warmer schedule, or the `P2-15` augmentation issue fixed first. | `val_acc` clearly above `100/num_classes`% on a real run, with the log excerpt in a handoff. |
| `P2-15` | **P2** | `ImageNetSubsetDataset` hardcodes `transforms.Resize(256)` for **both** train and eval, then `RandomCrop(image_size)` / `CenterCrop(image_size)`. At 96 px the crop keeps `(96/256)² = 14%` of the area. `_check_aug.py`: **both** the train and the deterministic eval arm failed to memorise 89 real images in 80 steps, while synthetic noise memorised fine. Top open convergence suspect. | `Resize` scales to `max(image_size, 256)` (or an explicit `--resize_size`); the 89-image memorisation probe reaches ~100% train accuracy; result in a handoff. Related: `P3-7`. |
| `P2-16` | **P2** | GoogLeNet aux heads spatially collapse below ~128 px: `InceptionOutClassifier`'s `AvgPool2d(kernel_size=5, stride=3)` yields `feature_map_size` 1/2/2/4 at 96/128/160/224 px. At 1 the aux branch is crippled, so any small-image probe of aux behaviour is invalid. | Construction raises (or warns loudly) when `image_size` makes the aux feature map < 2; probes documented to use ≥160 px. |
| `P2-17` | **P2** | Convergence assertions must use `1.6·ln(K)` for GoogLeNet's combined loss (`main + 0.3·aux1 + 0.3·aux2`), not `ln(K)`. Comparing the combined loss to `ln(K)` mislabels a healthy network as broken — this cost real debugging time in HANDOFF_003. | A shared helper returns the chance threshold per network; every probe/test uses it. |
| `P2-18` | **P2** | Label/image alignment through the *real* sampler is unverified. `_check_labels2.py` was written but never executed. | Probe run and its output in a handoff; mismatch count is zero. |
| `P3-11` | **P3** | Parquet resume correctness unverified. `_check_resume.py` (kill after 90 s, then compare label/path fingerprints against an uninterrupted reference) was written but never executed. | Probe passes: resumed subset is byte-identical in labels/paths to the uninterrupted one. |
| `P3-12` | **P3** | Decide the fate of `--shuffle_buffer` now that the parquet path ignores it (row groups are already globally shuffled). Currently dead weight except on the streaming fallback — **and actively harmful**: `shuffle_buffer_size` is part of the cache `expected_metadata` comparison but *not* of `_cache_key`, so changing `--shuffle_buffer` re-downloads a byte-identical subset ("Cache metadata mismatch, re-downloading..."). | Either documented as streaming-fallback-only, or removed; if kept, exclude it from the metadata equality check (or include it in the cache key) so it cannot invalidate the cache. Decision recorded in a handoff. |
| `P3-13` | **P3** | `train_final.py` `CONFIG` values are **informed defaults, not measured**. They must be re-baked from a real GPU run before anyone cites them as tuned. | `CONFIG` numbers match a completed GPU run that beat chance; handoff records the measured run. |

