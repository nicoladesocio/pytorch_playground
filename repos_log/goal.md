# Repository Goal (curated)

> Single source of truth for *why* this repo exists and what "done" means.
> Keep this file curated: update it in the same PR that changes direction.
> Working state lives in [KANBAN.md](./KANBAN.md); agent-to-agent narrative lives in
> [handoffs/](./handoffs/).
>
> **New agent? Start at [AGENT_INSTRUCTIONS.md](./AGENT_INSTRUCTIONS.md)** — it defines the
> reading order, the PR workflow, and how to run the code. Read only the **latest** handoff,
> never all of them.

## Mission

Train the CNN architectures that already exist in this repository — **AlexNet** and
**GoogLeNet** — on an ImageNet-1k subset, using **PyTorch + PyTorch Lightning**, and fix
the issues that currently prevent them from training correctly and reproducibly.

This is a *playground*: the point is working, inspectable, reproducible training runs and
clean reference implementations of the papers — not state-of-the-art accuracy.

## Scope

**In scope**

- `AlexNet/` — AlexNet (Krizhevsky et al., 2012) reference implementation + training.
- `GoogLeNet/` — GoogLeNet (Szegedy et al., 2014) with Inception modules + auxiliary heads.
- `trainer/` — the **active** model-registration + training CLI. This is the supported entrypoint.
- `utils/` — dataset caching/subsetting, checkpoint discovery, shared helpers.
- `custum_datasets/` — dataset wrappers (note: folder name is misspelled; renaming is tracked
  on the kanban, not done silently).
- Making training runs reproducible: seeds, recorded configs, recorded metrics.
- **One parameterless "final" script per network** (`AlexNet/train_final.py`,
  `GoogLeNet/train_final.py`) with every best-known hyperparameter baked in — see
  *Deliverable: parameterless final scripts* below.

**Out of scope (unless explicitly promoted on the kanban)**

- New architectures beyond AlexNet/GoogLeNet.
- `trainer_v2/` — it is a **design draft with non-working hardcoded registries**. Finish it
  only as a scoped kanban task; do not treat it as the training path.
- Full ImageNet (1.2M images). We train on a reproducible subset (default 51 200 train /
  2 048 val, 64 classes).
- Distributed / multi-GPU training.

## Training size policy (explicit user requirement)

**Trainings must stay small.** They have to be runnable on a PC with a *bigger* GPU than the
dev box, and the whole point of the playground is that a run finishes in a reasonable time.
Concretely:

- Never default to full ImageNet. The canonical subset is 51 200 train / 2 048 val samples
  over 64 classes (`utils/datasets_utils.DEFAULT_*`).
- Anything used for *validation by an agent* must be a smoke run: a few hundred to a few
  thousand samples, 1–3 epochs, CPU-friendly.
- Long/full runs are the human's job, on the bigger GPU, via the final scripts.

## Deliverable: parameterless final scripts (explicit user requirement)

When a network is considered ready, the agent authors **one script per neural network** that
takes **no parameters at all** and has every best hyperparameter baked into it. The human
then runs exactly that script to do the real training.

- `python AlexNet/train_final.py` and `python GoogLeNet/train_final.py` — no arguments
  required, no editing needed.
- Each script holds a single `CONFIG` dict that is the *recorded best* configuration, and it
  delegates to the same `main()` as `train.py` so there is exactly one training engine.
- `--smoke` is the only accepted argument: a CPU-sized variant used by agents/CI to prove the
  pipeline works. It is *not* the deliverable and must never be what the human runs.
- The baked-in values must be justified by a log excerpt in a handoff. A number in a final
  script that was never measured is a bug.

## Definition of done

A model is considered **trained** when all of the following hold:

1. `python trainer train <Model>` runs end-to-end from a clean checkout with no manual
   code edits.
2. Training loss **decreases below random-guess level** and validation accuracy rises
   clearly above chance for the chosen number of classes
   (chance for 64 classes = `log(64) ≈ 4.16` loss / `1.56 %` accuracy).
3. The run is **reproducible**: same seed + same config ⇒ comparable metrics.
4. Artifacts land in deterministic, per-run paths: checkpoints, TensorBoard logs,
   resolved config, and a metrics summary.
5. No per-batch `print()` spam; observability comes from Lightning loggers.
6. A short run (smoke) and a real run both complete without OOM on the target hardware.
7. `python <Model>/train_final.py` exists, takes no arguments, and reproduces the metrics
   recorded in the handoff that justified its `CONFIG`.

## Current status (convergence updated as of HANDOFF_005)

"Trainable?" = produces a **converged**, reproducible run, not merely a run that starts.
Both models execute `Trainer.fit()` on CPU without error (HANDOFF_002).

| Model | Trainable? | Notes |
|---|---|---|
| GoogLeNet | **Yes — converged once** | `val_acc` 4.58 % → 34.6 % against a 1.56 % chance level, `val_loss` 4.01 → 2.84, 64 classes, 227 px, GPU ([E030](./evidences/E030-googlenet-converged-run.md)). Needs `--train_samples_per_epoch 16384`; a default epoch is 65 steps and looks flat. |
| AlexNet | **Yes — converged once, with Adam** | `val_acc` 4.58 % → 24.8 %, `val_loss` 4.03 → 3.11 ([E031](./evidences/E031-alexnet-adam-converges-sgd-pinned-at-chance.md)). Its SGD arm stays at chance. E031 attributed that to `gradient_clip_val=1.0` scaling every step by 0.037 against an init gradient norm of 26.9; **that attribution has since been refuted** — clip off and clip=30 are also exactly at chance, so the clip is not the blocker (`P2-25`, [E033](./evidences/E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)). Eight candidate causes are now eliminated by measurement — clip, fp16, step budget, init, schedule, gradient spread, dropout, `weight_decay` — across 19 SGD configurations ([E033](./evidences/E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)–[E038](./evidences/E038-per-group-lr-sgd-sweep.md)). `train_acc` never exceeds 1.95 % in any of them, so SGD fails to fit even its own training stream; the surviving suspect is the 4 153-unique-image effective training set. |

Both models have now produced a verified converged run with committed evidence, which is the
half of "Trainable?" that had been open since HANDOFF_001. What is **not** yet met is the rest of
the definition of done above: the converged runs were stopped early (epoch 36 / 12 of 60) so there
is no full-length checkpoint, the `CONFIG` throughput numbers are unmeasured at the batch sizes
shipped (`P2-26`), and no run has been made reproducible end-to-end from `train_final.py` on the
target hardware.

**Validation resolution (`P2-27`, resolved).** `create_eval_sampler` covers every in-class
validation sample exactly once, so `val_acc` is quantised in steps of `100 / (in-class val
images)`. The canonical `val_samples=2048` cache holds only **153** in-class images, i.e. a
resolution of **0.654 %** — coarser than most effects this repo has drawn conclusions from, and
the reason HANDOFF_003 read 1–7 images changing side as training "oscillation". The documented fix
is **`val_samples=16384`**, which yields **1030 in-class validation images and a resolution of
0.0971 %** (measured by `python probes/check_val_pool.py --val_samples 16384 --predict 1000`,
[E036](./evidences/E036-validation-pool-size-and-accuracy-resolution.md)); both `CONFIG` dicts now
ship that value. One caveat travels with it: `val_samples` feeds `choose_target_classes`'s
`min_val_count` filter, so changing it changes *which* 64 classes are chosen — 8 of 64 differ
between the two pools. Accuracy figures from the 2048 caches (E030, E031) are therefore **not**
comparable with runs on the 16384 pool; re-baseline rather than comparing across. See
[KANBAN.md](./KANBAN.md) for the prioritised work.

## Ground rules for agents

1. **One handoff per work session.** Append `handoffs/HANDOFF_<NNN>.md` (progressive number,
   never reuse a number) describing what you did, what you found, and what you left open.
2. **Move kanban cards, don't invent new ones** for existing work. Add a card if work is
   genuinely new; link it to the handoff that created it.
3. **Update this file** if the mission, scope, or definition of done changes. Say so in your
   handoff.
4. **Never commit** credentials, `nscacert.crt`, `Data/datasets/**`, or `*.ckpt` — they are
   gitignored for a reason.
5. **Prefer `trainer/` over `trainer_v2/`** for any real training run.
6. **Evidence over assertion.** Claim a model "trains" only with a log/metric excerpt in
   your handoff.
7. **Work on a feature branch and open a PR; never merge to `main` yourself.** The agent's
   deliverable is *handoff + PR*; the human reviews and merges. This applies to code **and**
   documentation. If you cannot open a PR (no GitHub credential — see
   [AGENT_INSTRUCTIONS.md §6](./AGENT_INSTRUCTIONS.md)), push the branch and record the exact
   PR title/body in your handoff. **Do not merge instead.**
   *(The direct merge in HANDOFF_001 was a deviation and is not a precedent.)*

## Environment reality check

**Hardware facts do not travel between agents.** This section was previously a list of one
machine's findings written as repo-wide rules, and it wrongly concluded the GPU was
unusable. It is now a *procedure*: measure first, then record your host in the table.

**First command of every session:**

```bash
python probes/check_hardware.py     # verdict: GPU_OK | CPU_ONLY | GPU_BROKEN | HOST_ONLY
```

Read [AGENT_INSTRUCTIONS.md §3–§4](./AGENT_INSTRUCTIONS.md) for how to get to `GPU_OK`.
The image is built from `Dockerfile.gpu` in this repo, so it works on GPU and CPU-only
hosts alike and needs no vendor registry.

- **GPU works wherever the driver and wheel arch list allow it.** On the RTX A1000 6GB
  Laptop GPU (driver 535.309.01, CUDA 12.2, `sm_86`) in `ppg-gpu:latest`, `torch 2.10.0+cu128`
  reports `cuda avail True` and a real conv forward+backward through both networks passes.
  The old "GPU is unusable" finding was specific to a WSL2 host whose driver (465.89)
  predates CUDA passthrough and whose GTX 960M is `sm_50`, which modern wheels dropped.
  Neither is a property of this repo.
- **CPU works and is the fallback.** Both `AlexNetLightning` and `GoogLeNetLightning`
  complete a real `Trainer.fit()` on CPU; the HANDOFF_003 §6.8 numbers reproduce
  bit-exactly on this host. Use small `--train_samples` / `--batch_size` / `max_epochs`.
- **`--shm-size` matters.** Docker's default `/dev/shm` is 64 MB; a multi-worker DataLoader
  carrying image batches exhausts it and dies with
  `unable to allocate shared memory ... No space left on device`. Start the container with
  `--shm-size=8g`. `utils.runtime.count_available_workers` clamps `num_workers` to what shm
  can actually support, so the failure degrades to `num_workers=0` instead of crashing.
- **Memory cap matters.** The default Docker Desktop limit (8 GB) is not enough for
  streaming-based dataset downloads; the process is OOM-killed. The parquet row-group path in
  `utils/datasets_utils.py` needs only ~0.5 GB, so this is no longer a blocker, but raising
  the limit still helps.
- `pyproject.toml` is **not** the GPU problem — it declares plain `torch`/`torchvision` and
  pip resolves a working build per machine. On a CUDA host the blocker, if any, is the host
  driver or the wheel's compiled arch list; `check_hardware.py` tells you which.
- Default paths are repo-relative via `utils.runtime.repo_root()`, so runs work in-container
  and on the host (`P1-4` closed).

Plan training work on whatever `check_hardware.py` reports. Do not assume either device
based on this document.
