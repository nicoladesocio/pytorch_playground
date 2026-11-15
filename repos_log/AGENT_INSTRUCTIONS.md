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
| 5 | `evidences/` — **only the files a card links to** | Raw proof: command, host, output. Read one file, not the folder |

**Do not read the whole `handoffs/` directory.** If you genuinely need older history, `grep`
for a specific card ID rather than reading files end to end. The same applies to `evidences/`:
it is a lookup table keyed by card, not a narrative — open the specific `E<NNN>` file a card
cites and nothing else.

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

### 2.2 Branch and PR naming

- Branch: `agent/<short-topic>` (e.g. `agent/fix-alexnet-main-guard`).
- PR title: `<Card-ID>: <imperative summary>` — e.g. `P1-1: Move AlexNet Trainer block under __main__ guard`.
- PR body must list: card IDs addressed, what was verified, **how** it was verified (command +
  output excerpt), and what is knowingly left undone.
- One PR per coherent card group. Do not bundle unrelated fixes.

### 2.3 Handoff file rules

- Progressive number, **never reuse a number**, never edit an older handoff.
- Write the handoff bit by bit, so that even if your session is prematurely terminated, you have something written down
- Record failures and dead ends too — they save the next agent hours.
- State your **assumptions** explicitly if you resolved any ambiguity yourself.
- The handoff is for **narrative and decisions**. Raw proof goes in `evidences/` (§2.4); the
  handoff links to it rather than being the only place it exists.

### 2.4 Evidence files — `repos_log/evidences/`

Every claim that a card is done must be checkable **without reading a handoff**. A handoff is
600–800 lines; if a `DONE` row says "Evidence: HANDOFF_004 §6.4", the next agent has to load the
whole document to verify one number, and the number is on page six. So evidence gets its own file.

- **One evidence piece = one file.** Never merge two experiments into one file.
- **ID is `E` + three digits, append-only.** Never renumber, never reuse a retired ID; new evidence
  takes the next free number. Filename `E<NNN>-<kebab-case-summary>.md`.
- **Add a row to [`evidences/INDEX.md`](./evidences/INDEX.md)** — the index is how anyone finds
  evidence by card instead of by guesswork.
- **Every file starts with the same metadata table:** Status, Cards, Command, Host, First recorded.
  The `Command` must be copy-pasteable. If you cannot reproduce it, it is an anecdote, not evidence.
- **Raw output goes in fenced blocks verbatim.** Interpretation goes *below* the block, never inside
  it. Never fill in a number before the run has finished — if you correct one later, add a
  correction note rather than silently editing.
- **Superseded evidence stays**, marked `SUPERSEDED` in **Status**, with a pointer to what replaced
  it and why the old claim was wrong. Deleting it destroys the record of how the repo was misled
  (`E004`/`E005` are kept precisely for that reason).
- **A `DONE` card links its evidence file in the kanban's Evidence column.** A `DONE` row whose
  evidence link is broken, or which cites only a handoff section, is not done.
- State what the evidence does **not** show. `E021` proves the GPU path *executes*; it does not
  prove convergence. Most arguments in this repo have gone wrong at exactly this seam.

---

## 3. Environment — how to actually run the code

**Everything runs in Docker, and the GPU question is answered by measurement, never by this
document.**

> **The single most important rule in this file:** agents run on *different machines*. What is
> true about the hardware on one host is not true on another. §4 explains why, and
> [`probes/check_hardware.py`](../probes/check_hardware.py) is how you find out where you are.
> **Run the probe before you decide anything about CPU vs GPU.**

### 3.0 First command of every session — the hardware probe

```bash
# from the repo root (the folder containing pyproject.toml)
python3 probes/check_hardware.py --host-only     # host: driver, docker, nvidia toolkit, shm
```

It prints `PASS`/`FAIL`/`WARN` per property and a verdict. Do not skip it: an earlier agent
spent days training on CPU because a *previous* agent's "GPU is unusable" note had been
promoted into a standing rule in this file, on a machine that had a working GPU.

### 3.1 The working container — build it from `Dockerfile.gpu`

`Dockerfile.gpu` is the portable, hardware-agnostic image: no corporate cert, no `code-server`,
no vendor registry, and it is the same image on a GPU box and a CPU-only box.

```bash
docker build -f Dockerfile.gpu -t ppg-gpu:latest .

docker run -d --name ppg-dev --gpus all --shm-size=8g --entrypoint bash \
  -v "$PWD:/workspace" -w /workspace \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ppg-gpu:latest -c "sleep infinity"
```

Then run everything with `docker exec ppg-dev bash -lc 'cd /workspace && ...'`.

Verified on the host this was authored on (Ubuntu 22.04, i7-13800H, 62 GB RAM,
RTX A1000 6GB Laptop): `torch 2.10.0+cu128`, `torchvision 0.25.0+cu128`, `lightning 2.6.6`,
`datasets 5.2.0`, `torchmetrics 1.9.0`, `pyarrow 26.0.1`, Python 3.12 — and
`torch.cuda.is_available() == True` with `sm_86`.

Why each flag is there:

| Flag | Why |
|---|---|
| `--gpus all` | GPU passthrough. Requires `nvidia-container-toolkit` on the host; the probe tells you. Harmless on a CPU-only host. |
| `--shm-size=8g` | Docker's 64 MB default `/dev/shm` kills multi-worker DataLoaders with `unable to allocate shared memory ... No space left on device`. |
| `-v ~/.cache/huggingface:...` | `ILSVRC/imagenet-1k` is gated. Mounting the cache means the token `huggingface-cli login` already wrote is used, so the secret never appears in a command line or a file. |
| bind mount | The repo source is bind-mounted, so host edits are live inside the container. **You never rebuild the image to iterate on Python code** — only when `pyproject.toml` changes. |

`Dockerfile.gpu` ends with an import gate for every declared dependency, so a missing dep fails
the *build* rather than surfacing 40 minutes into a training run. (That is exactly how the old
image ended up "built" without `lightning` in it — kanban `P2-12`.)

### 3.2 Gotchas that will otherwise waste an hour

1. **The repo's `Dockerfile` does not build.** It `COPY`s the gitignored `nscacert.crt` and
   installs EOL `openjdk-11-jdk` (kanban `P1-5`). Use `Dockerfile.gpu`. The
   `nicoladesocio/pytorchplayground:latest` image some handoffs reference is a third,
   unrelated artefact: its `ENTRYPOINT` is `code-server` (so `docker run <image> python3 ...`
   dies with `exec: "code-server": executable file not found in $PATH`) and it is missing most
   of the declared deps. If you are ever stuck with it, pass `--entrypoint bash` and
   `pip install --break-system-packages lightning datasets tensorboard scipy pandas scikit-learn tqdm`.
2. **`pip install` inside the container needs `PIP_BREAK_SYSTEM_PACKAGES=1`.** The 2.10 CUDA
   images dropped conda and use Debian's Python, which carries a PEP 668
   `EXTERNALLY-MANAGED` marker. `Dockerfile.gpu` sets it, so plain `pip install` works.
   Note `.devcontainer/devcontainer.json` still points at a nonexistent
   `/opt/conda/bin/python` — stale, ignore it.
3. **`pip install .` would shadow the bind mount.** `Dockerfile.gpu` therefore uninstalls the
   project and reinstalls it editable, so `import utils` resolves to `/workspace/utils`. If you
   ever build a variant image and host edits appear to be ignored, this is why.

### 3.3 If Docker is not available on your host — the venv fallback

Docker is preferred (it is how every run in the handoffs was produced, and it pins the CUDA
wheels), but a plain venv is a legitimate fallback on a host where you cannot install the
container toolkit. The only hard requirements are Python ≥ 3.10 and a `torch` build whose
compiled arch list covers your GPU (`probes/check_hardware.py` checks exactly this).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
# CPU-only host: add --index-url https://download.pytorch.org/whl/cpu to shrink the download
pip install torch torchvision          # let pip pick the build for your machine
pip install -e .
python probes/check_hardware.py        # must say GPU_OK or CPU_ONLY before you train
```

If pip installs a CUDA build that does not cover your GPU's `sm_XX`, install the matching
wheel from <https://download.pytorch.org/whl> (cu121 / cu124 / cu128 …) or the CPU index.

### 3.4 Running the training CLI

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

### 3.5 Importing model modules

`GoogLeNet/model.py` uses `from inception import ...`, a bare top-level import that only
resolves when `GoogLeNet/` itself is on `sys.path`. When probing from the repo root, set both:

```bash
PYTHONPATH=/workspace:/workspace/GoogLeNet python -c "import model"
```

**Do not `import model` from both model folders in one process.** `AlexNet/model.py` and
`GoogLeNet/model.py` are both named `model.py` and both are imported as bare `import model`,
so the second import silently returns the first module — you get
`AttributeError: module 'model' has no attribute 'AlexNet'` and conclude the model is broken
when only your import was. Load them from explicit paths under distinct module names;
`probes/check_hardware.py::_load_model_module` is the working pattern.

---

## 4. GPU vs CPU — measure, do not trust this document

**There is no standing answer to this question. Run `python probes/check_hardware.py` inside
your container and believe its verdict.**

### 4.1 Why the old rule was wrong

Until HANDOFF_004 this section read:

> "**Experiments must run on CPU. Do not spend time trying to enable the GPU.**"

That was a correct description of **one specific machine** (WSL2, driver 465.89, GTX 960M
Maxwell `sm_50`) written as if it were a property of the repository. It was later run on a
native-Ubuntu laptop with an RTX A1000 6GB (Ampere `sm_86`, driver 535.309.01) where
`docker run --gpus all` works first try and `torch.cuda.is_available()` is `True`. The
"don't bother with the GPU" rule was actively costing agents days of CPU training.

The lesson is not "the GPU works" — it is that **hardware facts do not travel between agents.**
Any statement in this repo of the form "X is impossible on the hardware" is a claim about one
host at one point in time, and must be re-measured.

### 4.2 The four things that actually decide GPU usability

| Property | How to check | Failure mode |
|---|---|---|
| Driver new enough for the CUDA runtime | `nvidia-smi` driver ≥ 525 for CUDA 12.x | `nvidia-smi` works but CUDA init fails |
| `nvidia-container-toolkit` installed | `nvidia-ctk --version` | `docker run --gpus all` cannot work at all |
| Wheel compiled for your arch | `torch.cuda.get_arch_list()` contains your `sm_XX` | every kernel launch fails, however new the driver |
| Real conv forward+backward on device | `probes/check_hardware.py` | matmul passes, cuDNN conv dies |

The third row is the generalisation of the `sm_50` conclusion. It is the *wheel's* arch list
that made the GTX 960M unusable, not CUDA in the abstract — a `sm_86` card on the same wheel is
fine. Checking `get_arch_list()` against `get_device_capability()` tells you in one line which
side of that line you are on.

The fourth row matters because a driver/wheel mismatch commonly passes a matmul smoke test
(cuBLAS is precompiled) and dies on the first cuDNN convolution. Only a real forward+backward
through the repo's own networks proves the GPU is usable.

### 4.3 What to do with the verdict

| Verdict | Meaning | Action |
|---|---|---|
| `GPU_OK` | CUDA verified on both networks | Run the smoke tests **on GPU**. Do not fall back to CPU. |
| `CPU_ONLY` | no GPU visible to `nvidia-smi` | Run the smoke tests on CPU; say so in your handoff. |
| `GPU_BROKEN` | a GPU exists but torch cannot use it | **Fixable.** Chase it before falling back to CPU; record findings. |
| `HOST_ONLY` | you ran `--host-only` | Inconclusive by construction; re-run inside the container. |

Record the probe's full output in your handoff, near the top, before any training numbers. It
is what makes your results comparable with another agent's on different silicon.

### 4.4 Sizing on a GPU

The `CONFIG` in both `train_final.py` targets a ≥10 GB card (`16-mixed`, batch 96/128 at
227 px). On a 6 GB card that OOMs — halve the batch and keep the rest of the recipe. `--smoke`
sizes (1024/256 samples, 2 epochs, batch 16) fit comfortably on 6 GB and are what an agent
should run. `sm_80+` also gets `bf16-mixed`, which is more numerically stable than `16-mixed`
for these nets; if `16-mixed` produces NaN losses, try `bf16-mixed` before suspecting the model.

### 4.5 What is still true

`pyproject.toml` declares plain `torch`/`torchvision`, so pip resolves a working build per
machine — the `.toml` was never the GPU problem. And CPU remains a valid target: both models
complete a `Trainer.fit()` on CPU, so a CPU-only agent is not blocked, merely slower.

---

## 5. Working rules

1. **Evidence over assertion — and evidence lives in its own file.** Never claim a model
   "trains" without the command and the raw output that prove it. Put that in a file under
   `repos_log/evidences/` (§2.4), not in your handoff: a `DONE` card that cites only
   `HANDOFF_004 §6.4` forces the next agent to read an ~800-line document to check one number.
   Run it; paste the output; link the file from the card's **Evidence** column.
2. **Reproduce before you fix.** Confirm the bug exists, then fix it, then confirm it is gone.
3. **Move kanban cards; don't invent duplicates.** `KANBAN.md` has **exactly four
   chapters — `TODO`, `IN PROGRESS`, `DONE`, `BACKLOG / IDEAS` — and no others.** Do not add a
   `REVIEW / WAITING` chapter, and do not append a `### Added by HANDOFF_<NNN>` sub-table;
   those existed and were removed in HANDOFF_004 because they let cards drift out of the
   board's reach. New work → a row in `TODO` in ID order with a `P1/P2/P3` prefix; never
   renumber existing cards. `BACKLOG / IDEAS` is for unscheduled ideas, not cards.
4. **An `IN PROGRESS` card must name what is still missing.** Appending an "update" to a card
   is not progress and is not a way to close it. Either its acceptance criteria are met and
   the evidence file exists (move it to `DONE`), or they are not — in which case the card
   carries an explicit **Still missing:** clause. A card parked in `IN PROGRESS` with no gap
   stated will be read as nearly-done by the next agent, and it usually is not.
5. **`BACKLOG / IDEAS` holds only unscheduled work.** It is a list of things still to do, not a
   changelog. When an idea (or half of one) is finished, it leaves the backlog: give it an ID and
   put it in `DONE` with its evidence, and either delete the bullet or rewrite it so that what
   remains is *only* the unfinished part, pointing at the `DONE` row. Never leave a completed or
   struck-through (`~~like this~~`) item sitting in the ideas list — a reader scanning the backlog
   for work to pick up cannot tell done from not-done, and half-done bullets quietly become
   permanent. (`P2-22` and `P2-23` are the corrected examples.)
6. **Never commit** credentials, `nscacert.crt`, `Data/datasets/**`, or `*.ckpt` (gitignored).
7. **Prefer `trainer/` over `trainer_v2/`.** `trainer_v2/` is a non-working draft.
8. **Don't rename `custum_datasets/`** silently — it is a tracked card; a repo-wide rename
   buries the real diff.
9. **Clean up.** Delete probe scripts, scratch files, and containers you started
   (`docker rm -f <name>`) before opening the PR. Check `git status` for stray files.
10. **Small, reviewable PRs.** The human reviews these by hand. A 40-file PR gets rejected;
   a 3-file PR with evidence gets merged.
11. **If blocked, stop and document.** Make a reasonable assumption, write it down, continue.
   If you truly cannot proceed, leave the card `TODO` with a comment in your handoff rather
   than half-fixing it.

---

## 6. Git / GitHub — the procedure is portable, the credentials are not

**Read this section's warning first: everything in it was written by an agent on a
particular machine, and "it works here" does not travel.** §4 had exactly this failure —
a handoff recorded one host's auth routes as repo-wide rules, and the next agent on
different hardware wasted a session rediscovering that none of them existed. So this
section is split: a procedure that is the same everywhere, and a per-host table that you
*append to* rather than rewrite.

### 6.1 The procedure (same on every host)

1. Work on a branch named per §2.2; never commit to `main`.
2. Find out who you actually are before believing anything below:
   ```bash
   git remote -v
   gh auth status                      # identity + which token, if gh is installed
   env -u GH_TOKEN -u GITHUB_TOKEN gh auth status   # same, ignoring env-injected tokens
   ssh -T -o BatchMode=yes git@github.com   # does an SSH key work?
   ```
   **An environment token can shadow a working keyring login.** `GH_TOKEN` / `GITHUB_TOKEN`
   take precedence over `gh`'s stored credentials, so `gh auth status` may report a *different,
   weaker* identity than the one you would actually push with. If the env identity is read-only
   but the keyring identity is not, the env token is the whole problem — retry with it unset
   before concluding you cannot push. Compare the two `gh auth status` lines above; if they name
   different users, you have this case.
3. Test push permission *without* pushing:
   ```bash
   git push --dry-run origin HEAD:<branch>
   ```
   A `403 Permission denied` here is decisive — it means the identity behind your
   credentials cannot write, and no amount of remote-juggling fixes it. Do not loop
   through remotes hoping one works.
4. If the dry run is clean, push and open the PR.
5. If it is not, **push is not available to you.** Do the fallback in §6.3. Never merge
   to `main` yourself, and never ask for a merge — that is the human's call.

### 6.2 Per-host auth facts

Append your host; do not delete other rows, and do not promote your row into a rule.

| Host | Identity | Push | Notes |
|---|---|---|---|
| WSL2, GTX 960M (HANDOFF_001–003) | `nicoladesocio` | ✅ via SSH | HTTPS push rejected (`Password authentication is not supported`), so push through a temporary SSH remote. `gh` not installed. GitKraken MCP `pull_request_create` worked. |
| `LX0350043`, RTX A1000 (HANDOFF_004) | `F93194C_stla` via `GH_TOKEN`, **but `nicoladesocio` in the gh keyring** | ✅ once the env token is unset | `gh` v2.4.0 is installed. The injected `GH_TOKEN` identity is read-only (`push: false`) and shadows the keyring login, so the first push 403s; `env -u GH_TOKEN git push` succeeds as `nicoladesocio`. `ssh -T` → `Permission denied (publickey)`; forking is blocked (Enterprise Managed User). GitKraken MCP is not available in this agent's toolset. |

Two traps that recur across hosts:

- **A `--force-with-lease` push needs a local remote-tracking ref.** Run
  `git fetch <remote> <branch>` first, or it fails with
  `! [rejected] ... (stale info)` — which reads like a conflict but is not one.
- **`gh auth status` succeeding does not mean you can push.** It reports the token's
  *identity*, not its *permissions*. Use the `--dry-run` push in §6.1 step 3.

### 6.3 Fallback when you cannot push

Push nothing. In your handoff, write a `## PR to open` section containing the exact
branch name, PR title, complete PR body, and the
`https://github.com/nicoladesocio/pytorch_playground/compare/main...<branch>?expand=1`
URL, so the human can open it in two clicks. Then say so plainly in your final message —
do not imply the PR exists.

### 6.4 Repo root

The repo root may be **nested** (on the HANDOFF_001–003 hosts it was
`pytorch_playground/pytorch_playground`; the outer folder was only a container). Confirm
it rather than assuming:

```bash
git rev-parse --show-toplevel
```

---

## 7. Quick reference

Portable form — run from the repo root reported by `git rev-parse --show-toplevel`.

```bash
REPO=$(git rev-parse --show-toplevel) && cd "$REPO"

# 0. hardware probe — FIRST command of every session (§3.0)
python probes/check_hardware.py

# 1. build the image once (§3.1). Works on GPU and CPU-only hosts.
docker build -f Dockerfile.gpu -t ppg-gpu:latest .

# 2. start the dev container
docker run -d --name ppg-dev --gpus all --shm-size=8g \
  -v "$REPO:/workspace" -w /workspace \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ppg-gpu:latest -c "sleep infinity"

# 3. the two parameterless deliverables (§3.4)
docker exec ppg-dev bash -lc 'cd /workspace && python AlexNet/train_final.py --smoke'
docker exec ppg-dev bash -lc 'cd /workspace && python GoogLeNet/train_final.py --smoke'

# 4. probes
docker exec ppg-dev bash -lc 'cd /workspace && python probes/check_hardware.py'
docker exec ppg-dev bash -lc 'cd /workspace && python probes/check_overfit2.py --device cuda'
docker exec ppg-dev bash -lc 'cd /workspace && python probes/check_labels2.py'
docker exec ppg-dev bash -lc 'cd /workspace && python probes/check_resume.py'

# 5. trainer CLI (registry.json is gitignored -> register first)
docker exec ppg-dev bash -lc 'cd /workspace && python -m trainer register --model-name GoogLeNet'
docker exec ppg-dev bash -lc 'cd /workspace && python -m trainer train GoogLeNet --dry-run'

# 6. clean up (§5.7)
docker rm -f ppg-dev
```

If Docker is unavailable, §3.3 has the venv fallback.

**Card IDs** (`P1-1`, `P2-3`, …) are stable. Reference them in commits, PR titles, PR bodies,
and handoffs.
