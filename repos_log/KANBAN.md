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
| — | *nothing in flight* | | |

---

## 👀 REVIEW / WAITING

| ID | Card | Blocked on | Owner |
|---|---|---|---|
| `P2-11` | AlexNet convergence (vanishing gradient, loss stuck at 6.9) | Needs `P1-1`,`P1-2`,`P1-4`,`P2-1` done + CUDA box to iterate | — |
| `P3-6` | Decide fate of `trainer_v2/` (finish vs. delete vs. freeze) | Product decision from repo owner | — |

---

## ✅ DONE

| ID | Card | Done in | Handoff |
|---|---|---|---|
| `P0-1` | Explore repo; audit good / bad / missing | HANDOFF_001 | [001](./handoffs/HANDOFF_001.md) |
| `P0-2` | Create `repos_log/` with curated `goal.md` + `handoffs/` | HANDOFF_001 | [001](./handoffs/HANDOFF_001.md) |
| `P0-3` | Create this shared kanban | HANDOFF_001 | [001](./handoffs/HANDOFF_001.md) |

---

## 📋 TODO — P1 (blocks every training run)

| ID | Card | Acceptance criteria | Est. |
|---|---|---|---|
| `P1-1` | **Fix `AlexNet/train.py`: `Trainer` block escapes `__main__` guard.** Lines 73–93 (`trainer = pl.Trainer(...)` … `save_model_checkpoint(...)`) are module-level, so they execute on *import* — including `import model`-adjacent imports and any test collection. `lightning_model`/`args` are only bound inside the guard. | `python -c "import AlexNet.train"` is a no-op; whole body indented under `__main__`; smoke train completes. | S |
| `P1-2` | **Declare `pytorch_lightning` dependency.** `GoogLeNet/model.py` + `GoogLeNet/train.py` import `pytorch_lightning`, but `pyproject.toml` lists only `lightning`. `lightning` does not reliably install the legacy `pytorch_lightning` top-level package. | Standardise **all** imports on `lightning` (preferred) *or* add `pytorch_lightning` to deps; `python -c "import GoogLeNet.train"` succeeds in a clean venv. | S |
| `P1-3` | **Fix `trainer` CLI invocation.** `python trainer …` and `python trainer/trainer.py …` (both documented in `README.md` / `trainer/README.md`) crash with `ModuleNotFoundError: No module named 'trainer.model_base'; 'trainer' is not a package` — the script's own folder lands on `sys.path[0]` and shadows the `trainer` package. Only `python -m trainer` works. | All three documented forms work, or docs + `trainer/README.md` state `python -m trainer` as the only supported form and the script-form fallback is removed. Add a smoke test. | S |
| `P1-4` | **Portable paths.** Every default is container-only (`/workspace/AlexNet/checkpoints`, `/workspace/Data/datasets/`, `/workspace/GoogLeNet/logs`). Nothing runs outside the devcontainer. | Paths derive from `PROJECT_ROOT` (or a `--data-root` / `PYTORCH_PLAYGROUND_ROOT` env var); defaults resolve correctly on Linux, in-container, and on Windows. | M |
| `P1-5` | **`Dockerfile` build is broken.** `COPY nscacert.crt …` references a gitignored, absent file → `docker build` fails at step 1. Also pins `openjdk-11-jdk` (EOL) purely for a corporate cert. | Image builds from a clean checkout; corporate-cert handling is opt-in (build arg / separate stage), not mandatory. | M |
| `P1-6` | **`Data/datasets/` placeholder missing.** `.gitignore` whitelists `!Data/datasets/.gitkeep` but the directory doesn't exist, so the documented cache root has no anchor and `load_dataset_subset` must `makedirs` blind. | `Data/datasets/.gitkeep` committed; cache root exists after clone. | XS |

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
- **GPU strategy for this host**: document a CPU-only smoke profile (tiny samples, batch 8) so agents without CUDA can still validate plumbing.
