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

**Out of scope (unless explicitly promoted on the kanban)**

- New architectures beyond AlexNet/GoogLeNet.
- `trainer_v2/` — it is a **design draft with non-working hardcoded registries**. Finish it
  only as a scoped kanban task; do not treat it as the training path.
- Full ImageNet (1.2M images). We train on a reproducible subset (default 51 200 train /
  2 048 val, 64 classes).
- Distributed / multi-GPU training.

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

## Current status (as of HANDOFF_002)

"Trainable?" = produces a **converged**, reproducible run, not merely a run that starts.
Both models execute `Trainer.fit()` on CPU without error (HANDOFF_002); neither has converged.

| Model | Trainable? | Notes |
|---|---|---|
| GoogLeNet | Partially | Most stable path. Completes `Trainer.fit()` on CPU (HANDOFF_002). Uses bare top-level imports (`from inception import …`) that break package-style imports. |
| AlexNet | No | Runs on CPU, but plateaus at `loss ≈ 6.9` (= `log(1000)`, random guessing) with vanishing gradients. Also has a `Trainer` block sitting outside its `__main__` guard, and a latent `256*6*6` classifier shape bug (`P2-13`). |

Neither model has ever produced a verified, reproducible converged run with committed
evidence. See [KANBAN.md](./KANBAN.md) for the prioritised work.

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

**Everything runs in Docker; experiments run on CPU.** Verified in HANDOFF_002:

- The working container is `nicoladesocio/pytorchplayground:latest` (already built locally,
  33.5 GB). It has a `code-server` `ENTRYPOINT`, so you must pass `--entrypoint` to
  `docker run`, and it is **missing the repo's declared deps** (`lightning`, `datasets`, …) —
  install them once per container.
- **GPU is unusable.** `docker run --gpus all` fails with
  `nvidia-container-cli: initialization error: ... libnvidia-ml.so.1: file too short`, and
  `torch.cuda.is_available()` is `False`. Two independent causes: the host driver (465.89)
  predates WSL2 CUDA passthrough, and the GTX 960M is Maxwell `sm_50`, no longer supported by
  modern PyTorch wheels.
- **CPU works.** Both `AlexNetLightning` and `GoogLeNetLightning` complete a real
  `Trainer.fit()` on CPU. Use small `--train_samples` / `--batch_size` / `max_epochs`.
- `pyproject.toml` is **not** the GPU problem — it declares plain `torch`/`torchvision` and
  pip resolves a working build per machine. The blocker is the host driver.
- All default paths are still container-only (`/workspace/...`), so nothing runs on the host
  without path fixes (kanban `P1-4`).

Plan training work against the container on **CPU**. Treat a CUDA machine as an optional
accelerator, not a prerequisite.
