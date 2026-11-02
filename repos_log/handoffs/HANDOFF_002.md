# HANDOFF_002 — Collaboration contract + verified Docker/CPU environment

**Agent:** second agent (session 2)
**Date:** 2026-11-02
**Branch:** `agent/handoff-002-env-and-instructions`
**Cards:** `P0-4`, `P0-5` done · `P1-2` corrected · `P1-7`, `P2-12`, `P2-13` added · board hygiene
**Read next:** only this file + [KANBAN.md](../KANBAN.md) + [goal.md](../goal.md). Do **not** read HANDOFF_001 unless you need a specific prior claim.

---

## 1. Why this session exists

Two requests from the repo owner:

1. **Define how agents and the human collaborate.** Specifically: stop merging to `main`;
   the agent's deliverable should be *handoff + PR*, and the human reviews/merges. Also
   produce an **agent instructions** file so a freshly spawned agent knows to read the goal and
   **only the latest handoff** (to avoid burning context).
2. **Answer, empirically: can the code run on GPU using `pyproject.toml`? If not, on CPU?**
   The owner assumed the `.toml` should still work on the old GPU. Experiments are essential to
   the repo, so a runnable path is mandatory.

Mid-session the owner recalled the repo was "supposed to run inside Docker" and asked me to
build the Dockerfile. That redirect turned out to be the productive path — see §3.

---

## 2. What I did (step log)

| # | Action | Result |
|---|---|---|
| 1 | Refreshed the stale `origin/main` ref over SSH | `acced71..87a9f2f`; `main` == `origin/main`; cleared HANDOFF_001's misleading "ahead 3" |
| 2 | Probed host: disk 295 GB free, Python 3.13.14, `uv`/`gh`/`gk`/`conda` **missing**, `docker` present | Only Docker is usable |
| 3 | Confirmed `nscacert.crt` absent + gitignored; read `devcontainer.json` | Repo `Dockerfile` cannot build as-is (`P1-5`) |
| 4 | `docker info` → daemon 29.2.0, overlayfs, 8 CPUs | Docker usable |
| 5 | `docker images` → **`nicoladesocio/pytorchplayground:latest` already exists, 33.5 GB** | Owner had already built it; no need to rebuild |
| 6 | `docker manifest inspect pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` | Tag exists upstream but is **not cached locally** |
| 7 | `docker run <image> python3 -c ...` | ❌ `exec: "code-server": executable file not found in $PATH` — image has a `code-server` ENTRYPOINT |
| 8 | Same with `--entrypoint python3` | ✅ Python 3.12.3, `torch 2.13.0+cu132`, `cuda avail False` |
| 9 | `docker run --gpus all ...` | ❌ `nvidia-container-cli: initialization error: load library failed: /usr/lib/wsl/drivers/.../libnvidia-ml.so.1: file too short` |
| 10 | Checked installed dists in the image | `lightning`, `pytorch_lightning`, `datasets`, `tensorboard`, `scipy`, `pandas`, `scikit-learn`, `tqdm` **all MISSING**; only `torch`/`torchvision`/`numpy` present |
| 11 | `wsl --version` / `wsl -l -v` | WSL 2.4.10, kernel 5.15.167.4, only `docker-desktop` distro |
| 12 | Host driver check | `465.89`, GTX 960M, 4 GB |
| 13 | Simulated the Dockerfile's `COPY pyproject.toml .` + `pip install .` in a clean container | ✅ **succeeds** — builds a wheel. So the layer doesn't fail; it silently installs a project without the source tree |
| 14 | Started persistent container `ppg-smoke` with repo bind-mounted at `/workspace` | ✅ |
| 15 | `pip install lightning datasets tensorboard scipy pandas scikit-learn tqdm` | ✅ (only a harmless `spin`/`click` conflict warning) |
| 16 | Ran an env probe script | All deps present; **both `import lightning` and `import pytorch_lightning` work** (2.6.6) |
| 17 | First model-import attempt from `/workspace/_probe/` | ❌ `No module named 'AlexNet'` — my own bug: `sys.path[0]` was the script's folder |
| 18 | Re-ran with `PYTHONPATH=/workspace` | `AlexNet.model` ✅ · `GoogLeNet.model` ❌ `No module named 'inception'` · `GoogLeNet.inception` ✅ |
| 19 | Forward/backward smoke at 224×224 | ❌ **`RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x6400 and 9216x4096)`** |
| 20 | Swept input sizes through `AlexNet.features` | `224→6400` ✗ · `227→9216` ✓ · `231→9216` ✓ · `256→9216` ✓ |
| 21 | Checked what the dataset actually emits | `ImageNetSubsetDataset` defaults to `image_size=224`, **but both `train.py` pass `image_size=227`** → bug is latent, not live |
| 22 | Emulated script-mode imports (`sys.path[0]=GoogLeNet`) | `import model` ✅ and `from inception import ...` ✅ — so GoogLeNet works in its intended invocation |
| 23 | **Real `Trainer.fit()` on CPU, both models, 227px, synthetic data** | ✅ **AlexNet 58.3 M params — `max_epochs=2` reached** · ✅ **GoogLeNet 10.3 M params — `max_epochs=2` reached** |
| 24 | Probed `pytorch_lightning` provenance | `lightning 2.6.6` **requires `pytorch-lightning`** → HANDOFF_001's `P1-2` premise was wrong |
| 25 | Built the wheel and listed contents | `trainer/` **absent** from the wheel → new card `P1-7` |
| 26 | Read `AlexNet/test_synthetic.py` | Already covers 227px + `.cuda()` (matches existing `P2-8`) |
| 27 | Cleaned up: deleted `_probe/`, `_wheel_probe.py`, `_pl_probe.py`; `docker rm -f ppg-smoke` | `git status` clean of scratch files |
| 28 | Wrote `AGENT_INSTRUCTIONS.md`; corrected `goal.md`; updated `KANBAN.md`; wrote this file | — |
| 29 | Verified the documented runbook end-to-end in a fresh container | `python -m trainer train GoogLeNet --dry-run` **fails on a fresh checkout** — `registry.json` is gitignored, so `register` must run first. Runbook corrected. |
| 30 | Cleaned up: `docker rm -f ppg-verify`, deleted `trainer/registry.json` and all probe scripts | `git status` shows only the four `repos_log/` files |

---

## 3. The GPU/CPU answer (the owner's actual question)

### Short answer

**No, not on GPU — but the `.toml` is not the problem, and yes, CPU works.** Run experiments on CPU.

### `pyproject.toml` is fine

It declares plain `torch` and `torchvision` with no pin, so pip resolves a build appropriate to
whatever machine installs it. Nothing in the `.toml` prevents GPU use. The owner's assumption
was reasonable; the blocker is elsewhere.

### GPU is blocked for two independent reasons

1. **The host driver (465.89) predates WSL2 CUDA GPU passthrough.** Docker Desktop runs
   containers inside WSL2, and WSL2 GPU support requires a much newer driver. The container
   cannot even load `libnvidia-ml.so.1`:
   ```
   nvidia-container-cli: initialization error: load library failed:
   /usr/lib/wsl/drivers/nvacui.inf_amd64_.../libnvidia-ml.so.1: file too short
   ```
2. **The GPU is Maxwell `sm_50` (GTX 960M, 4 GB).** Current PyTorch CUDA wheels no longer ship
   `sm_50` kernels. Even with a working driver, the prebuilt wheels would not accelerate it.

Fixing (1) is a human action (update the NVIDIA driver on the host). Fixing (2) is not possible
on this hardware with modern PyTorch — it would need a source build with `sm_50` enabled, which
is not worth the effort for this repo.

### CPU works — this is the supported experiment path

Both models completed a genuine Lightning `fit()` on CPU inside the container:

```
0 | net | AlexNetLightning | 58.3 M | train
`Trainer.fit` stopped: `max_epochs=2` reached.
  AlexNet    fit() OK  (227px, CPU)

0 | net | GoogLeNetLightning | 10.3 M | train
`Trainer.fit` stopped: `max_epochs=2` reached.
  GoogLeNet  fit() OK  (227px, CPU)
```

So **no card should be blocked on "needs a CUDA box" anymore.** I removed that blocker from
`P2-11`. Use small `--train_samples` / `--batch_size` / `max_epochs` on CPU.

### The image situation (important, and surprising)

`nicoladesocio/pytorchplayground:latest` (33.5 GB) exists locally, but **it was not built from
the `Dockerfile` in this repo**:

| | Repo `Dockerfile` | Image on disk |
|---|---|---|
| Base | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` | PyTorch **2.13.0** + CUDA **13.2.1** |
| Python | (base default) | 3.12.3 |
| `code-server` | no | **yes**, as `ENTRYPOINT` |
| `nscacert.crt` | required (build fails without it) | corporate-cert layers absent |

Practical consequences for every agent:

- `docker run <image> python3 ...` **fails** — you must pass `--entrypoint`.
- The image is **missing most declared dependencies**. Install once per container:
  `pip install --break-system-packages lightning datasets tensorboard scipy pandas scikit-learn tqdm`.
- The repo source is **bind-mounted**, so you never need to rebuild the image to iterate on
  Python code. This is why the drift hasn't bitten yet — and why it's easy to miss.

I did **not** rebuild the image. Rebuilding from the repo `Dockerfile` would (a) fail on the
missing `nscacert.crt`, and (b) replace a working 33.5 GB environment with a 2.10.0-based one
for no benefit. Tracked as `P2-12`.

---

## 4. New findings

### `P1-7` — `pip install -e .` does not install the `trainer` CLI

`pyproject.toml`'s hatchling `packages` list is `["AlexNet", "GoogLeNet", "utils", "custum_datasets", "trainer_v2"]`
— **`trainer/` is missing**. Verified by building the wheel; its contents include `AlexNet/*`,
`GoogLeNet/*`, `utils/*`, `custum_datasets/*`, `trainer_v2/*` and **no `trainer/`**.

So the only working entrypoint (`python -m trainer`) works exclusively from a source checkout.
XS to fix.

### `P2-13` — AlexNet's classifier hardcodes `256*6*6` (latent crash)

`AlexNet/model.py:39` → `nn.Linear(256 * 6 * 6, 4096)`. But conv1 is `stride=4` with **no
padding**, so spatial size is `floor((H-11)/4)+1` then pooled:

| input | features | flat | vs. 9216 |
|---|---|---|---|
| 224 | `(1,256,5,5)` | 6400 | ✗ `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x6400 and 9216x4096)` |
| 227 | `(1,256,6,6)` | 9216 | ✓ |
| 231 | `(1,256,6,6)` | 9216 | ✓ |
| 256 | `(1,256,6,6)` | 9216 | ✓ |

**Why this is not currently breaking training:** both `AlexNet/train.py:59-60` and
`GoogLeNet/train.py:55-56` explicitly pass `image_size=227`. But `ImageNetSubsetDataset`'s own
default is `image_size=224`, so any caller that omits the argument crashes. This is a landmine,
not a live fire. XS/S to fix properly.

### `P2-12` — repo `Dockerfile` ≠ image in use

See §3. Also note a subtle packaging trap I confirmed: `COPY pyproject.toml .` followed by
`pip install .` **succeeds silently** even when every `packages` directory is absent from the
build context. A Dockerfile can therefore "install dependencies" while installing nothing, and
the build stays green.

### `P1-2` — my predecessor's premise was wrong (corrected)

HANDOFF_001 claimed `lightning` "does not reliably provide the legacy `pytorch_lightning`
top-level package" and recommended declaring `pytorch_lightning`. **Verified false:**
`lightning 2.6.6` declares `Requires-Dist: pytorch-lightning`, and installing `lightning`
installs `pytorch_lightning 2.6.6` as a real distribution. Both import styles work.

The *real* remaining problem is different and still worth fixing: `GoogLeNet/model.py:4` uses
`from inception import ...` and `GoogLeNet/train.py:4` uses `import model` — bare top-level
imports that only resolve when `GoogLeNet/` is itself on `sys.path`. From the repo root,
`import GoogLeNet.model` fails with `ModuleNotFoundError: No module named 'inception'`.
`GoogLeNet/` also has no `__init__.py` (already `P3-2`). I rewrote the card accordingly rather
than deleting it.

### Board hygiene

`P3-6` (`trainer_v2/` fate) appeared **twice** — once in REVIEW/WAITING and once in the P3
list. I removed the REVIEW duplicate and left a pointer. Card count is now 35 with no duplicate
IDs.

---

## 5. Files changed this session

| File | Change |
|---|---|
| `repos_log/AGENT_INSTRUCTIONS.md` | **new** — reading order, PR-as-deliverable contract, Docker runbook, CPU policy, git/auth notes |
| `repos_log/goal.md` | Points to instructions file; ground rule 7 rewritten (no self-merge, carve-out removed); environment section replaced with verified facts |
| `repos_log/KANBAN.md` | `P0-4`/`P0-5` done; `P1-2` corrected; `P1-7`/`P2-12`/`P2-13` added; `P2-11` unblocked from CUDA; `P3-6` duplicate removed |
| `repos_log/handoffs/HANDOFF_002.md` | **new** — this file |

**No production code was changed.** All findings are recorded as cards for the next agent.

---

## 6. Deviation record

**I did not merge to `main` this session**, per the new contract. HANDOFF_001 merged directly;
that carve-out is now removed from `goal.md` ground rule 7 and explicitly disavowed in
`AGENT_INSTRUCTIONS.md` §2.1.

**I could not open the PR myself.** I tried four routes and all failed for lack of credentials:

| Route | Result |
| --- | --- |
| `gh` CLI | not installed |
| GitKraken CLI (`gk`) MCP `pull_request_create` | `The prompt was cancelled or did not authorize the action` — run `gk auth login` to fix |
| Browser → GitHub compare view | served the diff but the session is **signed out** ("Sign in" in the header) |
| HTTPS `git push` | rejected (`credential.helper = manager-core`, non-interactive) |

Only **SSH** authenticates (as `nicoladesocio`), so I pushed the branch over a temporary
`origin-ssh` remote and removed it afterwards. The PR therefore needs one manual click:

<https://github.com/nicoladesocio/pytorch_playground/compare/main...agent/handoff-002-env-and-instructions?expand=1>

The title and body to paste are below. **Do not merge it yourself if you are an agent** —
merging is the human's step (`AGENT_INSTRUCTIONS.md` §2).

> If a future agent installs and authenticates `gh`, `gh pr create` becomes the one-line
> replacement for this whole section. That is worth doing once and would remove this friction
> permanently.

## PR to open

- **Branch:** `agent/handoff-002-env-and-instructions` → `main`
- **Title:** `P0-4/P0-5: Add agent instructions, PR workflow contract, verified Docker/CPU runbook`
- **Body:**
  - Establishes the collaboration contract: agent delivers **handoff + PR**; human reviews and merges. Removes the docs-only direct-merge carve-out from HANDOFF_001.
  - Adds `repos_log/AGENT_INSTRUCTIONS.md` (read goal → **latest handoff only** → kanban).
  - Answers the GPU question empirically: GPU unusable (driver 465.89 predates WSL2 CUDA passthrough + `sm_50` unsupported); **CPU works** — both models complete `Trainer.fit()`.
  - Corrects `P1-2` (`lightning` *does* provide `pytorch_lightning`).
  - Adds `P1-7` (`trainer/` missing from wheel), `P2-12` (Dockerfile ≠ image in use), `P2-13` (AlexNet `256*6*6` latent shape bug).
  - Removes duplicate `P3-6`.
  - Docs-only; no production code changed.

---

## 7. Recommended next actions

1. **`P1-7`** (XS) — add `trainer` to `packages`. One line, unblocks real installation.
2. **`P2-9`** (M) — pytest + ruff + CPU smoke CI. Highest leverage: turns every audited bug
   into something a machine catches, and CPU now provably works so CI is feasible.
3. **`P1-1`** (S) — indent the `AlexNet/train.py` `Trainer` block under `__main__`.
4. **`P1-3`** (S) — fix or document the `trainer` CLI invocation forms.
5. **`P1-2`** (S) — standardise Lightning imports; make GoogLeNet's intra-package imports
   package-relative.
6. **`P2-13`** (S) — derive AlexNet's flatten size instead of hardcoding `256*6*6`.
7. **`P2-11`** — AlexNet convergence, **on CPU**, starting with the overfit-one-batch
   experiment. No longer blocked on hardware.

### Open questions for the owner

- **Driver upgrade?** Updating the NVIDIA driver is the only route to GPU use. Worth it, or is
  CPU sufficient for this playground?
- **Which image is canonical?** Keep the drifted `nicoladesocio/pytorchplayground:latest`
  (torch 2.13 + code-server) and fix the repo `Dockerfile` to match it, or rebuild from the repo
  `Dockerfile` (torch 2.10, no code-server)? This decides `P2-12` and `P1-5`.
- **`nscacert.crt`** — is the corporate cert still needed at all? It blocks `docker build`.
- Still unanswered from HANDOFF_001: fate of `trainer_v2/` (`P3-6`), and whether gated
  `imagenet-1k` is accessible or we should switch to `imagenette`.

---

## 8. Deliberately did NOT do

- **Did not rebuild the Docker image** — would fail on `nscacert.crt` and downgrade a working
  environment. Filed as `P2-12`.
- **Did not upgrade the NVIDIA driver** — host-level change, needs owner consent.
- **Did not change production code** — findings are cards; each needs a real run to verify a fix.
- **Did not read HANDOFF_001 in full** while working, per the context rule I just wrote. I read
  it only to correct the `P1-2` claim and to record the deviation.
- **Did not install PyTorch on the host** — pointless without a usable GPU; the container is the
  environment.
- **Did not merge.**
