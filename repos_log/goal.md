# Repository Goal (curated)

> Single source of truth for *why* this repo exists and what "done" means.
> Keep this file curated: update it in the same PR that changes direction.
> Working state lives in [KANBAN.md](./KANBAN.md); agent-to-agent narrative lives in
> [handoffs/](./handoffs/).

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

## Current status (as of HANDOFF_001)

| Model | Trainable? | Notes |
|---|---|---|
| GoogLeNet | Partially | Most stable path. Imports `pytorch_lightning`, which is **not** a declared dependency. |
| AlexNet | No | Plateaus at `loss ≈ 6.9` (= `log(1000)`, random guessing) with vanishing gradients. Also has a `Trainer` block sitting outside its `__main__` guard. |

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
7. Work on a feature branch and open a PR; do not push to `main` directly.

## Environment reality check

The documented environment is the devcontainer (`pytorch/pytorch:2.10.0-cuda12.8`,
`/workspace` bind mount). The **Windows host is not a viable training box**:

- Host GPU is a GTX 960M (Maxwell, `sm_50`, 4 GB) on driver 465.89 → CUDA 11.3 era.
  Modern PyTorch CUDA wheels need a much newer driver, and `sm_50` is no longer supported.
- Host Python has `datasets`/`numpy`/`PIL` but **no `torch`, `torchvision`, or `lightning`**.
- All default paths are container-only (`/workspace/...`), so nothing runs on the host
  without path fixes (kanban: `portable-paths`).

Plan training work against the devcontainer (or another CUDA machine), and keep a
CPU-runnable smoke path for CI.
