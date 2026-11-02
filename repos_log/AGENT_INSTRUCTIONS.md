# Agent Instructions — how to work on this repo

> **Read this first, then [goal.md](./goal.md), then only the LATEST file in
> [handoffs/](./handoffs/).** Do **not** read every handoff — they are long and you will run
> out of context. Older handoffs are an archive; the kanban is the live state.

You are one of several agents working autonomously on this repository. The human reviews and
merges your work; they are not watching while you work. Be self-sufficient, but never
irreversible.

---

## 1. Required reading order (in this exact order)

| Step | File | Why |
|---|---|---|
| 1 | [AGENT_INSTRUCTIONS.md](./AGENT_INSTRUCTIONS.md) (this file) | How to work |
| 2 | [goal.md](./goal.md) | Mission, scope, definition of done, ground rules |
| 3 | [KANBAN.md](./KANBAN.md) | **Live** state — what is done, in flight, and free |
| 4 | `handoffs/HANDOFF_<NNN>.md` — **the highest number only** | Immediate context from your predecessor |

**Do not read the whole `handoffs/` directory.** If you genuinely need older history, `grep`
for a specific card ID rather than reading files end to end.

Before touching anything, confirm the card you picked is still `TODO` and unowned.

---

## 2. Definition of a finished agent session

A session is complete when **all** of these exist:

1. Code changes on a **feature branch** (never on `main`).
2. Kanban cards moved, with `Owner` and `Handoff` filled in.
3. A new `handoffs/HANDOFF_<NNN+1>.md`.
4. **An open Pull Request** for the human to review and merge.

### 2.1 The PR is the deliverable — never merge to `main` yourself

This is the collaboration contract with the human:

> **The agent produces a handoff + a PR. The human reviews and merges.**

- Push your branch and open a PR. **Stop there.**
- Do **not** merge, do **not** force-push to `main`, do **not** "temporarily" merge because a
  PR could not be opened.
- If you cannot open a PR (no credential — see §6), push the branch, state the exact PR
  command you would have run in your handoff, and leave the branch for the human.
  **Do not merge instead.**

> ⚠️ `HANDOFF_001` merged directly to `main`. That was a mistake and is **not** a precedent.
> The carve-out that allowed it has been removed.

### 2.2 Branch and PR naming

- Branch: `agent/<short-topic>` (e.g. `agent/fix-alexnet-main-guard`).
- PR title: `<Card-ID>: <imperative summary>` — e.g. `P1-1: Move AlexNet Trainer block under __main__ guard`.
- PR body must list: card IDs addressed, what was verified, **how** it was verified (command +
  output excerpt), and what is knowingly left undone.
- One PR per coherent card group. Do not bundle unrelated fixes.

### 2.3 Handoff file rules

- Progressive number, **never reuse a number**, never edit an older handoff.
- Include the *evidence*: the command you ran and the output that proves the card is done.
- Record failures and dead ends too — they save the next agent hours.
- State your **assumptions** explicitly if you resolved any ambiguity yourself.

---

## 3. Environment — how to actually run the code

**Everything runs in Docker. Do not try to train on the Windows host Python.**

The host has no `torch`/`lightning` and its GPU cannot run modern PyTorch (§4).

### 3.1 The working container

```bash
# from the repo root (the folder containing pyproject.toml)
docker run -d --name ppg-dev --entrypoint bash \
  -v "${PWD}:/workspace" -w /workspace \
  nicoladesocio/pytorchplayground:latest -c "sleep infinity"
```

Then run everything with `docker exec ppg-dev bash -lc 'cd /workspace && ...'`.

Verified inside that image: Python 3.12.3, `torch 2.13.0+cu132`, `torchvision 0.28.0+cu132`,
`lightning 2.6.6`, `pytorch_lightning 2.6.6`, `datasets`, `numpy`, `tensorboard`, `scipy`,
`pandas`, `scikit-learn`, `tqdm`.

### 3.2 Two gotchas that will otherwise waste an hour

1. **The image's `ENTRYPOINT` is `code-server`.** A plain `docker run <image> python3 ...`
   fails with `exec: "code-server": executable file not found in $PATH`. Always pass
   `--entrypoint` (bash or python3).
2. **The image does not contain the repo's declared dependencies.** It was built from a
   different Dockerfile than the one in the repo (it has `code-server`, and ships
   `torch 2.13.0+cu132` while the repo `Dockerfile` pins `2.10.0-cuda12.8`). If you start from
   a fresh container, install them once:
   ```bash
   pip install --break-system-packages lightning datasets tensorboard scipy pandas scikit-learn tqdm
   ```
   The repo source is **bind-mounted**, so edits on the host are live inside the container —
   you do not need to rebuild the image to iterate on Python code.

### 3.3 Running the training CLI

`trainer/registry.json` is **gitignored**, so a fresh checkout has no registered models and
`train` fails with `KeyError: "Model 'GoogLeNet' is not registered"`. Register first:

```bash
docker exec ppg-dev bash -lc 'cd /workspace && python -m trainer register --model-name GoogLeNet'
docker exec ppg-dev bash -lc 'cd /workspace && python -m trainer train GoogLeNet --dry-run'
```

Verified output of the second command:

```
Executing training command:
/usr/bin/python /workspace/GoogLeNet/train.py --version_name googlenet_v1 --num_classes 64 \
  --checkpoint_path /workspace/GoogLeNet/checkpoints --train_samples 51200 --val_samples 2048 --batch_size 64
```

Note the container-only `/workspace/...` paths leaking into the generated command (kanban
`P1-4`), and that `python -m trainer` is currently the **only** working form of that command
(kanban `P1-3`) — the `python trainer ...` and `python trainer/trainer.py ...` forms documented
in the READMEs crash.

### 3.4 Importing model modules

`GoogLeNet/model.py` uses `from inception import ...`, a bare top-level import that only
resolves when `GoogLeNet/` itself is on `sys.path`. When probing from the repo root, set both:

```bash
PYTHONPATH=/workspace:/workspace/GoogLeNet python -c "import model"
```

---

## 4. GPU vs CPU — the verified answer

**Experiments must run on CPU. Do not spend time trying to enable the GPU.**

Verified on this host (2026-11-02, HANDOFF_002):

| Check | Result |
|---|---|
| `docker run --gpus all ...` | ❌ `nvidia-container-cli: initialization error: load library failed: /usr/lib/wsl/drivers/.../libnvidia-ml.so.1: file too short` |
| `torch.cuda.is_available()` in container | `False`, device count `0` |
| Host driver | `465.89` — predates WSL2 CUDA GPU passthrough |
| Host GPU | GTX 960M, 4 GB, Maxwell `sm_50` — unsupported by current PyTorch wheels |
| `Trainer.fit()` on CPU, AlexNet + GoogLeNet | ✅ both complete |

The GPU is unreachable for **two independent reasons** (driver too old for WSL2 GPU
passthrough, and `sm_50` is no longer supported by modern PyTorch). Upgrading the driver is
the only path to GPU use and is a human action.

**Therefore: run smoke tests, convergence experiments, and CI on CPU.** Both models complete a
`Trainer.fit()` on CPU. Use a small `--train_samples` / `--batch_size` and a tiny
`max_epochs`. Only escalate to a CUDA machine if a card explicitly requires real ImageNet
throughput, and say so in your handoff.

`pyproject.toml` itself is fine — it declares plain `torch`/`torchvision`, and pip resolves a
working build for whatever machine installs it. The problem was never the `.toml`; it was the
host driver.

---

## 5. Working rules

1. **Evidence over assertion.** Never claim a model "trains" without a log/metric excerpt in
   your handoff. Run it; paste the output.
2. **Reproduce before you fix.** Confirm the bug exists, then fix it, then confirm it is gone.
3. **Move kanban cards; don't invent duplicates.** New work → new card at the bottom of
   `BACKLOG` with a `P1/P2/P3` prefix; never renumber existing cards.
4. **Never commit** credentials, `nscacert.crt`, `Data/datasets/**`, or `*.ckpt` (gitignored).
5. **Prefer `trainer/` over `trainer_v2/`.** `trainer_v2/` is a non-working draft.
6. **Don't rename `custum_datasets/`** silently — it is a tracked card; a repo-wide rename
   buries the real diff.
7. **Clean up.** Delete probe scripts, scratch files, and containers you started
   (`docker rm -f <name>`) before opening the PR. Check `git status` for stray files.
8. **Small, reviewable PRs.** The human reviews these by hand. A 40-file PR gets rejected;
   a 3-file PR with evidence gets merged.
9. **If blocked, stop and document.** Make a reasonable assumption, write it down, continue.
   If you truly cannot proceed, leave the card `TODO` with a comment in your handoff rather
   than half-fixing it.

---

## 6. Git / GitHub on this host

Read this before you waste time on auth.

- **HTTPS `git push` fails** — `Password authentication is not supported`.
- **SSH works** — `ssh -T git@github.com` → `Hi nicoladesocio!`.
  `origin` is HTTPS, so add a temporary remote to push:
  ```bash
  git remote add origin-ssh git@github.com:nicoladesocio/pytorch_playground.git
  git fetch origin-ssh <branch>          # needed before --force-with-lease
  git push --force-with-lease origin-ssh <branch>
  git remote remove origin-ssh
  ```
  A plain `--force-with-lease` without the preceding `fetch` fails with
  `! [rejected] ... (stale info)`, because the lease has no local remote-tracking ref to
  compare against.
- **Opening PRs: use the GitKraken MCP tool `pull_request_create`** with
  `provider: github`, `repository_name: pytorch_playground`,
  `repository_organization: nicoladesocio`. The GitKraken CLI is **signed in** on this host,
  so this works end-to-end. Verify a known PR with `pull_request_get_detail`.
- **Do not re-diagnose auth from scratch.** These routes are dead ends on this host — skip
  them:
  - `gh` CLI — not installed.
  - `git credential fill` — `fatal: Cannot prompt because user interactivity has been disabled`.
  - `gk auth login` — needs an interactive browser plus a pasted token; fails in a
    non-interactive shell with `stdin is closed and this prompt cannot be skipped`.
  - Integrated browser → GitHub — signed out; the compare view renders but has no session.
  - If `pull_request_create` reports `provider name is required`, you omitted `provider`.
    If it reports `did not authorize the action`, the GitKraken session expired — ask the
    human to sign in again rather than trying to script the login.
- **If PR creation genuinely fails**, push the branch and record the exact title, body, and
  `compare/main...<branch>?expand=1` URL in your handoff under `## PR to open`. **Still do
  not merge to `main` yourself.**

The repo root is **nested**: `pytorch_playground/pytorch_playground`. The outer folder is only
a container. `git` commands must run in the inner one.

---

## 7. Quick reference

```bash
cd d:/Users/Nicol/Documents/pytorch_playground/pytorch_playground   # real repo root

# start dev container
docker run -d --name ppg-dev --entrypoint bash -v "${PWD}:/workspace" -w /workspace \
  nicoladesocio/pytorchplayground:latest -c "sleep infinity"
docker exec ppg-dev bash -lc 'pip install --break-system-packages -q lightning datasets tensorboard scipy pandas scikit-learn tqdm'

# smoke test the CLI (registry.json is gitignored -> register first)
docker exec ppg-dev bash -lc 'cd /workspace && python -m trainer register --model-name GoogLeNet'
docker exec ppg-dev bash -lc 'cd /workspace && python -m trainer train GoogLeNet --dry-run'

# CPU training probe (models import only -- importing AlexNet.train starts a real run, see P1-1)
docker exec ppg-dev bash -lc 'cd /workspace && PYTHONPATH=/workspace:/workspace/GoogLeNet python -c "import model"'

# clean up
docker rm -f ppg-dev
```

**Card IDs** (`P1-1`, `P2-3`, …) are stable. Reference them in commits, PR titles, PR bodies,
and handoffs.
