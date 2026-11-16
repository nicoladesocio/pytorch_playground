# HANDOFF_005 — First converged runs in the repo's history; the augmentation probe turned out to be broken

**Agent:** fifth agent (session 5)
**Date:** 2026-11-15
**Host:** `LX0350043`, native Ubuntu 22.04.5, i7-13800H (20 cores), 62 GB RAM
**GPU:** NVIDIA RTX A1000 6GB Laptop GPU — Ampere `sm_86`, 6144 MiB, driver 535.309.01, CUDA 12.2
**Branch:** `agent/convergence-and-augmentation` (branched from `main` at `e246d77`, i.e. after PR #4 merged)
**Cards:** `P2-14`/`P2-11` converged runs delivered · `P2-21` extended to the train arm · `P2-15` reopened (its evidence was invalid) · new `P2-25`–`P2-29`
**Read next:** only this file + [KANBAN.md](../KANBAN.md) + [goal.md](../goal.md).

> Written incrementally, per `AGENT_INSTRUCTIONS.md` §2.3 — sections are filled as evidence lands.

---

## 0. TL;DR for the next agent

1. **Both networks now have a converged run — that is the headline, and it stands.** GoogLeNet
   `val_acc` 4.58 % → **34.6 %** (22× the 1.56 % chance level) and AlexNet → **24.8 %**, both with
   `--train_samples_per_epoch 16384`. These are real `train.py` runs against held-out validation and
   do not depend on the probe discussed in item 4. Evidence:
   [E030](../evidences/E030-googlenet-converged-run.md),
   [E031](../evidences/E031-alexnet-adam-converges-sgd-pinned-at-chance.md).
2. **`P2-15` is REOPENED, not closed.** I initially closed it on "4/4 PASS at 227 px"
   ([E029](../evidences/E029-augmentation-ab-both-arms-identical-images.md)), then found the A/B was
   invalid and marked E029 **SUPERSEDED**. Whether augmentation blocks learning is **again an open
   question**. Do not spend time re-deriving the converged runs; do spend it on `P2-15` at a sane
   `lr`.
3. **AlexNet's SGD arm is pinned at chance by `gradient_clip_val=1.0`, not by SGD.** Measured init
   total gradient norm **26.9** → every step scaled by 0.037, so `lr=0.1` behaves as `≈0.0037`. Adam
   is scale-invariant, which is the entire difference between the two AlexNet arms. No unclipped SGD
   run has been executed — `P2-25`.
4. **`probes/check_learn.py` had two RNG bugs, and its default `lr` makes its verdict unreliable.**
   (a) `main()` did not seed before `build_batch()`, and `list(sampler)` on a
   `WeightedRandomSampler` draws from the **global torch RNG** — so `--seed` never pinned the 32
   images and the same command line passed and failed on different invocations. (b) `build_batch()`
   was called **inside the arm loop**, so the eval and train arms memorised two *different* batches
   while the docstring claimed they were identical — the exact confound `--arm` exists to remove.
   Both are fixed (two concurrent identical runs now give bit-identical numbers). Then the `lr`
   sweep: same batch, same seed, same images at 227 px → `lr=1e-3` **16/16 PASS**, `lr=3e-3` (the
   default) **4/8**, `lr=1e-2` **2/16**. **Treat a single `check_learn` PASS/FAIL as noise.**
   Evidence: [E032](../evidences/E032-check-learn-probe-bugs-and-lr-sensitivity.md).
5. **The alarming "AlexNet cannot memorise 32 real images" reading is wrong.** It was an artefact of
   the probe's `lr`. At `lr=1e-3` AlexNet reaches `acc 0.969` eval / `0.812` train at 227 px.
6. **`loss 3.41 / acc 0.969` is not a contradiction and was not a bug.** On a 32-image batch one
   unfitted sample contributes its whole loss to the mean. The probe now prints per-sample
   `median`/`max` alongside the mean so an outlier-dominated mean is visible rather than mysterious.
   This finding came from E029 and **survives** its supersession.
7. **The actual convergence constraint is optimizer steps, not epochs.** The canonical subset holds
   51 200 rows but only **4 153** are in the 64 selected classes, so a default epoch is 65 steps at
   batch 64 and `max_epochs=60` is ~3 900 steps total. `--train_samples_per_epoch 16384` (drawn with
   replacement) is what made the difference. This is the mechanism behind `P2-10`'s "epoch size is
   misleading" and it is the single most useful knob for these runs.

---

## 1. Why this session exists

HANDOFF_004 §9 told the next agent, in order: (1) run the hardware probe, (2) take `P2-15` on the
GPU using `check_learn.py` as the harness and add an `--is_training` flag, (3) then `P2-14`/`P3-13`
with one real run that beats chance. I followed that order and did not re-litigate the GPU question.

---

## 2. Step log

| # | Action | Result |
|---|---|---|
| 1 | `probes/check_hardware.py --host-only`, then in-container | `PASS=7 FAIL=0 WARN=0`, `VERDICT: GPU_OK` — same host as HANDOFF_004 |
| 2 | Started `ppg-dev5` from existing `ppg-gpu:latest` | `--gpus all --shm-size=8g` |
| 3 | Added `--arm {eval,train,both}` + `--is_training` to `probes/check_learn.py` | train arm re-augments the same decoded PIL images every step |
| 4 | A/B at 227 px, smoke cache | 4/4 PASS — train arm memorises |
| 5 | A/B at 160 px, smoke cache | AlexNet train arm FAIL (`acc 0.469`) — size-dependent drag |
| 6 | Timed a real epoch on the canonical subset | 64 steps, 11 s — an epoch is tiny |
| 7 | Measured in-class counts on the canonical cache | 4 153 train / 153 val in the 64 chosen classes |
| 8 | A/B at 227 px, canonical cache | AlexNet eval cell reported `loss 3.41 acc 0.969 FAIL` — impossible-looking |
| 9 | Chased it: `torch.manual_seed` was after `make()` | unseeded init ⇒ same cell, different verdict per run |
| 10 | Fixed seeding, added median/max per-sample loss | same 4 cells all PASS, reproducibly |
| 11 | A/B at 160 px, both caches | 7/8 PASS; the one FAIL is AlexNet/train/1024-cache |
| 12 | Launched GoogLeNet convergence run | `val_acc` 5.88 % @ ep2 → 24.2 % @ ep10 → 34.6 % @ ep35, stopped at ep 36 |
| 13 | Launched AlexNet convergence run in parallel (SGD `lr=0.1`) | pinned at chance for 21 epochs (`val_loss 4.16 ≈ ln(64)`), killed |
| 14 | Measured AlexNet's init total gradient norm | **26.92** against `gradient_clip_val=1.0` → every step scaled by 0.037 |
| 15 | Relaunched AlexNet with adam `lr=3e-4` | `val_acc` 12.4 % @ ep2 → 24.8 % @ ep10, stopped at ep 12 |
| 16 | Gradient-clipping A/B micro-experiment | **CUDA OOM** — two training runs + a probe do not fit in 6 GB; see §7 |
| 17 | Baked the measured values into both `train_final.py` `CONFIG` dicts | `train_samples_per_epoch=16384`, AlexNet → adam/3e-4/cosine; smoke overrides reset to one pass |
| 18 | Added `--gradient_clip_val` to `AlexNet/train.py` | makes `P2-25` runnable without editing source |
| 19 | Re-ran both `--smoke` paths and validated both `CONFIG` dicts against `train.py`'s argparse | both pass on GPU; unknown-arg crash on the human's machine ruled out |
| 20 | Re-ran E029's documented command as a regression check | **different failing cell on each invocation** → the batch was never pinned by `--seed`; fixed by seeding at the top of `main()` |
| 21 | Two concurrent identical runs | **bit-identical** numbers (`2.9965 -> 2.4549 med 2.2906 max 5.432`) — proof the fix works |
| 22 | Found `build_batch()` was called per arm | the two arms had different batches → moved outside the arm loop; E029's A/B invalidated |
| 23 | Seed sweep + `lr` sweep at 227 px with the corrected probe | `lr=1e-3` 16/16 PASS, `3e-3` 4/8, `1e-2` 2/16 — the probe's own default decides the verdict; E029 marked SUPERSEDED, `P2-15` reopened, `P2-28`/`P2-29` opened |
| 24 | 160 px seed sweep (3 seeds, both caches) | 23/24 PASS; also measured that `--image-size` changes AlexNet's param count (30.0 M vs 58.3 M), so resolution and init are confounded |

---

## 3. Findings worth not re-discovering

### 3.1 `P2-15` was an untested hypothesis — and my test of it was also invalid

The card's own text said the train arm "has never actually been tested in isolation — it is still a
hypothesis, not a confirmed cause". That was accurate. I added `--arm`, ran the A/B, reported 4/4
PASS at 227 px and closed the card. Re-running my own documented command as a regression check later
in the same session showed the result did not reproduce, because the harness had two RNG defects
(§3.7, §3.8). E029 is now **SUPERSEDED** and `P2-15` is back in TODO.

The lesson is the same one `E016` and `E025` record — **a probe that can only fail one way is not a
probe**, and a card that has never been run is not a finding — plus one the repo did not have yet:
**a probe whose arms are not provably fed the same object is not an A/B.** A docstring asserting the
arms share a batch is not evidence that they do.

### 3.2 Seeding order is a correctness bug in a sweep harness, not a nicety

`torch.manual_seed(seed)` after `make()` seeds the *augmentation*, not the *network*. In a sweep
that reuses one process for eight cells, every cell's initialisation depended on how many RNG draws
the previous cells consumed. The symptom was a cell that passed and failed on repeat runs with an
identical command line — which is the worst possible failure mode because it looks like GPU
nondeterminism and invites you to go debug cuDNN.

### 3.3 Mean loss on a 32-image batch is an outlier statistic

`loss 3.41 / acc 0.969` reads as "the metric is broken". It is not: 31 samples at ~0 loss and one at
~109 gives exactly that. Any memorisation probe on a tiny batch must report the distribution, not
just the mean. The pass rule still uses the mean (it is the documented rule and what `goal.md`'s
definition of done is written against) — but the printed `median`/`max` are what you should actually
read.

### 3.4 An "epoch" on this subset is 4 153 samples, not 51 200

`choose_target_classes` picks the 64 best-represented classes out of 1000, and the canonical cache
holds ~65 images per class. So the in-class pool is 4 153 rows and the validation pool is **153
rows**. Consequences:

- `--train_samples 51200` describes the *download*, not the epoch. `--train_samples_per_epoch` is the
  knob that controls optimiser steps.
- `val_acc` is computed over 153 images, so its resolution is 1/153 = 0.65 %. A 24.2 % val_acc is
  ~37 images; it is a real signal but a coarse one. Any future card that wants a trustworthy
  validation number needs a bigger `--val_samples`, not a bigger model.
- `max_epochs=60` at the default epoch length is ~3 900 steps, which is why HANDOFF_003's 18-epoch
  AlexNet run "plateaued" — it had barely started.

### 3.5 A hardcoded default can decide an experiment's outcome

`gradient_clip_val=1.0` has been in `AlexNet/train.py` since before HANDOFF_001 and nobody measured
the gradient norm it clips against. Because Adam is scale-invariant, the clip is invisible in every
Adam run — so the repo ran dozens of Adam experiments and concluded "SGD doesn't work on AlexNet"
from the one arm where the clip dominates the update. **Before comparing two configurations, measure
what each default does to the update magnitude.** One `clip_grad_norm_(..., inf)` call would have
saved four handoffs of "AlexNet won't learn".

### 3.6 A probe is not a training run

The user's correction mid-session is worth repeating verbatim in spirit: the long runs exist to
answer *"is the network learning?"*, and the deliverable is the parameterless `train_final.py` the
human runs on a bigger GPU. GoogLeNet had answered the question by epoch 10; epochs 11–36 bought a
better checkpoint, not a better conclusion, and cost GPU time that a 6 GB card does not have to
spare. Both trainers already ship `EarlyStopping(monitor="val_loss", patience=max(10, max_epochs//5))`
— use it, and set `--max_epochs` to what the *question* needs rather than copying the deliverable's
value into a probe.

### 3.7 `WeightedRandomSampler` draws from the **global** torch RNG

Any probe that calls `list(sampler)` must seed *before* it, or the "fixed" batch is re-drawn on every
invocation and `--seed` is a lie. This is invisible in the code — nothing at the call site looks like
an RNG consumer — and its symptom (same command line, different failing cell) is identical to §3.2's
seeding-order bug and to GPU nondeterminism, so it will send you down the cuDNN path if you let it.
The fix is one `torch.manual_seed(args.seed)` at the top of `main()`, with a comment saying why.

### 3.8 An A/B harness must build its shared data **outside** the arm loop

`build_batch()` was called once per arm. Each call re-drew the sampler, so the eval arm and the train
arm trained on two different sets of 32 images while the docstring asserted they were the same. The
A/B measured nothing and still produced a clean, publishable 4/4 PASS. Self-consistency of comments
and docstrings is not evidence: for a shared-data A/B, the arms must be provably handed the *same
object*, and the cheapest way to guarantee that is to construct it once, above the loop.

### 3.9 A probe's default hyperparameter can be the dominant variable

At 227 px, the same batch/seed/images gave 16/16 PASS at `lr=1e-3`, 4/8 at the probe's default
`lr=3e-3`, and 2/16 at `lr=1e-2`. Nothing about the network or the data changed. Before attributing a
`check_learn` FAIL to the thing under study, run the baseline at two or three `lr` values and confirm
the baseline is comfortably *above* the pass threshold — otherwise you are measuring the probe.

Related confound: `--image-size` is also passed to the model constructor, so 160 vs 227 changes
`fc1`'s shape and the parameter-init draw (AlexNet 30.0 M vs 58.3 M params). Resolution, parameter
count and init stream are currently inseparable. That is `P2-28`.

---

## 4. Code changes

**New files**

| File | Purpose |
|---|---|
| `repos_log/evidences/E029-…md` | The `P2-15` A/B as originally run. Now marked **SUPERSEDED** with an in-file correction note; kept per the evidence contract. Its `run_cell()` seeding-bug finding and the `median`/`max` addition remain valid. |
| `repos_log/evidences/E030-…md` | GoogLeNet's converged run: full per-epoch table, copy-pasteable command, why the epoch length is the mechanism. |
| `repos_log/evidences/E031-…md` | AlexNet Adam-vs-SGD, with the measured init gradient norm and the clip factor. |
| `repos_log/evidences/E032-…md` | The two `check_learn.py` RNG bugs, the bit-identical reproducibility proof, the seed sweep, and the `lr` sweep that shows the probe's default `lr` dominates its verdict. Supersedes E029. |
| `repos_log/handoffs/HANDOFF_005.md` | This file. |

**Modified**

| File | Change |
|---|---|
| `probes/check_learn.py` | `--arm {eval,train,both}` and `--is_training`; `build_batch` returns `(pils, y, K, target_classes)` and is now called **once per cache, outside the arm loop** — it used to be called per arm, which re-drew the sampler and gave the two arms two different batches; new `make_arm_transform()` supplies each arm's transform from the real dataset class; `main()` **seeds before anything consumes the RNG** — `list(sampler)` on a `WeightedRandomSampler` draws from the global torch RNG, so without this `--seed` never pinned the batch; `run_cell` **seeds before `make()`** (and `cuda.manual_seed_all`) — previously the network's init inherited the previous cell's RNG state, so one command line produced both PASS and FAIL; per-sample `median`/`max` loss printed next to the mean, because a 32-image mean is an outlier statistic. Labels are computed via `class_to_idx` and were checked against `ds[i][1]`. |
| `AlexNet/train.py` | `--gradient_clip_val` is a real argument (default `1.0`, `<= 0` → `None`), replacing the hardcoded `gradient_clip_val=1.0`. Its help text states the measured init norm and why the default pins SGD at chance. Behaviour is unchanged unless the flag is passed. |
| `AlexNet/train_final.py` | `CONFIG`: `optimizer="adam"`, `lr=3e-4`, `scheduler="cosine"`, `train_samples_per_epoch=16384`, each with a comment naming the evidence file and, where relevant, the batch size it was measured at. `SMOKE_OVERRIDES`: `train_samples_per_epoch=None`. |
| `GoogLeNet/train_final.py` | `CONFIG`: `train_samples_per_epoch=16384` with the same justification; `SMOKE_OVERRIDES`: `train_samples_per_epoch=None`. |
| `repos_log/AGENT_INSTRUCTIONS.md` | §7 quick-reference `docker run` now includes `--entrypoint bash`, with a comment naming the `exec: "-c": executable file not found in $PATH` failure it prevents. Verified: without the flag the documented command fails exactly as described. |
| `repos_log/KANBAN.md` | `P2-14`, `P2-11` → ✅ DONE with evidence links; `P2-15` moved **back to TODO** with a rewritten card after its evidence was invalidated; `P3-13` rewritten with a **Still missing:** clause narrowed to the unmeasured throughput numbers; `P2-25`/`P2-26`/`P2-27`/`P2-28`/`P2-29` added to TODO in ID order. Still exactly four chapters. |
| `repos_log/goal.md` | Only the **"Current status"** section, per ground rule 3. It asserted "neither has converged" and "AlexNet: Trainable? No" — the exact claim this session produced counterexamples to, and the claim the next agent would otherwise inherit as fact. Both rows now cite E030/E031, and the paragraph below the table lists which parts of the definition of done are still unmet. **The definition of done itself was not changed.** |
| `repos_log/evidences/INDEX.md` | Rows for `E029`–`E032`; E029's Status changed to **SUPERSEDED** and E023's annotated as unverified (`P2-29`); the "Can these networks learn?" shortcut now points at the converged runs instead of saying `P2-14` is open; the "Is augmentation stopping learning?" shortcut now says unresolved; new shortcut on trusting a `check_learn` verdict. |

No dependency, packaging or CI files were touched.

---

## 5. How to reproduce this session

```bash
REPO=$(git rev-parse --show-toplevel) && cd "$REPO"

# 1. environment — expect VERDICT: GPU_OK on a CUDA host
python probes/check_hardware.py

docker run -d --name ppg-dev5 --gpus all --shm-size=8g \
  -v "$REPO:/workspace" -w /workspace \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint bash \
  ppg-gpu:latest -c "sleep infinity"

# 2. P2-15, the augmentation A/B (E029) — minutes, not hours
docker exec ppg-dev5 bash -lc 'cd /workspace && python probes/check_learn.py --steps 400 --arm both --image-size 227'
docker exec ppg-dev5 bash -lc 'cd /workspace && python probes/check_learn.py --steps 400 --arm both --image-size 160'

# 3. the converged runs (E030 / E031) — the exact commands are in each evidence file
docker exec ppg-dev5 bash -lc 'mkdir -p /tmp/ppglogs /tmp/ck_conv_goog /tmp/lg_conv &&
  nohup python GoogLeNet/train.py --version_name conv_goog_adam --num_classes 64 \
    --image_size 227 --train_samples 51200 --train_samples_per_epoch 16384 --val_samples 2048 \
    --batch_size 64 --eval_batch_size 128 --max_epochs 60 --optimizer adam --lr 1e-4 \
    --scheduler cosine --precision 16-mixed --accelerator gpu --num_workers 8 --seed 42 \
    --checkpoint_path /tmp/ck_conv_goog --log_path /tmp/lg_conv \
    > /tmp/ppglogs/goog_adam.log 2>&1 &'

# 4. the init gradient norm behind P2-25 (E031 has the full heredoc)
docker exec ppg-dev5 bash -lc 'cd /workspace && python - <<PY
import sys, torch, importlib.util
sys.path += ["AlexNet", "."]
s = importlib.util.spec_from_file_location("ppg_alex", "AlexNet/model.py")
m = importlib.util.module_from_spec(s); sys.modules["ppg_alex"] = m; s.loader.exec_module(m)
torch.manual_seed(42)
net = m.AlexNet(num_classes=64, image_size=227).cuda().train()
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(32,3,227,227, generator=g, device="cuda")
y = torch.randint(0,64,(32,), generator=g, device="cuda")
torch.nn.functional.cross_entropy(net(x), y).backward()
print("init grad norm", torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf")).item())
PY'

# 5. the deliverables still work (each ~1 min on GPU)
docker exec ppg-dev5 bash -lc 'cd /workspace && python AlexNet/train_final.py --smoke'
docker exec ppg-dev5 bash -lc 'cd /workspace && python GoogLeNet/train_final.py --smoke'
```

Expect: step 2 → **do not expect a fixed answer here.** At the probe's default `lr=3e-3` the 227 px
run currently gives 2/4 (both AlexNet cells fail); add `--lr 1e-3` and it gives 4/4. That sensitivity
*is* the finding — see §6.1 and [E032](../evidences/E032-check-learn-probe-bugs-and-lr-sensitivity.md).
step 3 → `val_acc` above 15 % within 8 epochs; step 4 → `init grad norm ≈ 27`; step 5 → RESULT
blocks with `device = cuda:0 (…)` and finite losses.

---

## 6. Evidence

### 6.1 `P2-15` — augmentation A/B: **not settled**

[E029](../evidences/E029-augmentation-ab-both-arms-identical-images.md) reported 4/4 PASS at 227 px
and closed the card. It is **SUPERSEDED**: the two arms were fed different batches and the batch was
never pinned by `--seed`. [E032](../evidences/E032-check-learn-probe-bugs-and-lr-sensitivity.md)
documents both bugs, proves the fixed probe is reproducible (two concurrent identical runs give
bit-identical numbers), and shows the probe's verdict is dominated by its own default `lr`
(`1e-3` → 16/16 PASS, `3e-3` → 4/8, `1e-2` → 2/16 on the same batch and seed).

What the corrected data *suggests* is that augmentation is not a blocker — at `lr=1e-3` both arms
memorise cleanly at 227 px, and 23/24 cells pass at 160 px — but that is not the controlled A/B the
card asks for, so `P2-15` stays open.

### 6.2 Convergence runs

Two runs, same host, same cache, same 227 px, both with `--train_samples_per_epoch 16384`. They ran
**in parallel** on the one 6 GB card (3.4 GB used).

| Run | Recipe | Stopped at | `val_acc` | `val_loss` | `train_acc` |
|---|---|---|---|---|---|
| GoogLeNet | adam `lr=1e-4`, cosine, batch 64, SWA on | ep 36 / 60 | 4.58 % → **34.6 %** | 4.01 → 2.84 | 2.2 → 48.5 |
| AlexNet (Adam) | adam `lr=3e-4`, cosine, batch 32 | ep 12 / 60 | 4.58 % → **24.8 %** | 4.03 → 3.11 | 2.67 → 24.5 |
| AlexNet (SGD) | sgd `lr=0.1`, cosine, batch 32 | ep 20 / 60 | 1.96 % (chance) | 4.16 ± 0.01 | 1.5 % (chance) |

Chance for 64 classes is 1.56 %; `ln(64) = 4.1589`; GoogLeNet's combined-loss chance value is
`1.6·ln(64) = 3.6841` ([E019](../evidences/E019-chance-level-per-network.md)). Full per-epoch
tables, copy-pasteable commands and the interpretation are in
[E030](../evidences/E030-googlenet-converged-run.md) (GoogLeNet) and
[E031](../evidences/E031-alexnet-adam-converges-sgd-pinned-at-chance.md) (AlexNet, both arms).

**Why they stopped early.** Both `train.py` files already install
`EarlyStopping(monitor="val_loss", patience=max(10, max_epochs//5))`, so a genuinely flat run stops
itself. Neither of these triggered it — they were stopped by hand once the question
*"is the network learning?"* had an unambiguous answer (GoogLeNet was at 19.6 % by epoch 10, AlexNet
at 18.3 % by epoch 8). The user's instruction for this session was explicit: these runs are probes
that the networks learn; the deliverable is the parameterless `train_final.py` the human runs on a
bigger GPU. Spending 60 epochs to confirm what epoch 10 already proved is not what the probes are
for. The 60-epoch length was only justified for the seeding-bug reproduction, where I did need
repeated identical runs.

**The SGD arm is a finding about clipping, not about SGD.** AlexNet's total gradient norm at
initialisation is **26.9** (measured on GPU, batch 32, 227 px); `train.py` passes
`gradient_clip_val=1.0`, so every step is scaled by **0.037** and `lr=0.1` behaves as `lr≈0.0037`.
Adam is scale-invariant, which is the entire difference between the two AlexNet arms. `--gradient_clip_val`
is now a real flag on `AlexNet/train.py` (`<= 0` disables clipping) but **no unclipped SGD run has
been executed** — that is `P2-25`, and until it is, the paper recipe is untested rather than
disproved.

### 6.3 What changed in the deliverables, and why each value is defensible

The deliverable is the two parameterless `train_final.py` scripts. Every value I baked in traces to
one of the runs above; everything I did *not* measure is marked as such in the file.

| Key | Before | After | Basis |
|---|---|---|---|
| `AlexNet.CONFIG["optimizer"]` | *(absent → SGD)* | `"adam"` | E031: SGD cannot move under the clip; Adam reached 24.8 % |
| `AlexNet.CONFIG["lr"]` | `0.01` | `3e-4` | E031, measured at batch 32 |
| `AlexNet.CONFIG["scheduler"]` | *(absent → step)* | `"cosine"` | E031's run used cosine |
| both `CONFIG["train_samples_per_epoch"]` | *(absent → 65-step epochs)* | `16384` | E030/E031; absent before, which is why every earlier run looked flat |
| both `SMOKE_OVERRIDES["train_samples_per_epoch"]` | *(inherited 16384)* | `None` | a smoke test must not become 1024 steps/epoch |
| `AlexNet/train.py --gradient_clip_val` | hardcoded `1.0` | flag, default `1.0` | makes `P2-25` runnable without editing source |

Deliberately **not** changed: `batch_size` (96 / 128) and `max_epochs`. The converged runs used 32
and 64 on a 6 GB card, so shipping 96/128 with the same `lr` is an unmeasured extrapolation — it is
flagged in-file and tracked as `P2-26` rather than quietly presented as validated. Both `--smoke`
paths were re-run after the change and still complete on GPU with finite losses, and both `CONFIG`
dicts were checked against `train.py`'s argparse so a typo cannot reach the human's run.

---

## 7. Environment notes for the next agent

Same as HANDOFF_004 §7, with these additions:

- `docker run` needs `--entrypoint bash` before `-c "sleep infinity"` on this image; the image's own
  `CMD` is not a shell, so `docker run ... ppg-gpu:latest -c "sleep infinity"` dies with
  `exec: "-c": executable file not found in $PATH`. The quick-reference block in
  `AGENT_INSTRUCTIONS.md` §7 omitted the flag and reproduced that error verbatim; it is fixed in this
  branch.
- Two `16-mixed` 227 px runs coexist on the 6 GB card: GoogLeNet at batch 64 (~1.8 GB) + AlexNet at
  batch 32 (~1.6 GB) = 3.4 GB used. Useful for A/B-ing recipes without serialising. **A third
  process OOMs** — I lost a gradient-clipping micro-experiment that way.
- Launch long runs with `docker exec ... bash -lc 'mkdir -p /tmp/ppglogs && nohup ... > /tmp/ppglogs/x.log 2>&1 &'`.
  `docker exec -d ... > /tmp/x.log` fails if the directory does not exist *inside* the container,
  because the redirect is evaluated by the host shell.
- To stop a run, read the PID (`ps -eo pid,args | grep "[t]rain.py"`) and `kill` it. `pkill -f
  "train.py --num_classes 64"` self-matches the `docker exec` shell's own command line and kills the
  wrapper instead, returning 143 while the training process survives.
- TensorBoard is available in the image, so `event_accumulator` is a legitimate way to read a run's
  scalars after the fact. Note `grad_norm/total` is logged in `on_after_backward`, i.e. **before**
  the fp16 loss scale is undone — divide by the current `GradScaler` scale (65536 early on) before
  comparing it to a directly measured norm.

---

## 8. Open / left for the next agent

New cards, all created from measurements in this session:

- **`P2-25` — unclipped SGD.** The single most valuable next run. `--gradient_clip_val 0` (or ~30)
  with `lr=0.1`, everything else as in E031's SGD arm. Until it exists, `train_final.py`'s Adam
  choice is "the recipe that worked", not "the recipe that beat SGD".
- **`P2-26` — `batch_size`/`lr` never measured at the shipped values.** Needs a ≥10 GB card, i.e.
  the human's machine. Either measure there or ship the measured batch sizes.
- **`P2-27` — validation is 153 images.** `val_acc` resolution is 0.654 %; every conclusion in this
  repo is drawn from ~150 images. Raise `--val_samples` until the in-class val pool is ≥1 000.
- **`P2-15` — reopened, and it is the other high-value run.** Re-run the eval/train A/B at a `lr`
  where the baseline is comfortably above the pass threshold (`--lr 1e-3` at 227 px per E032), ≥3
  seeds, both caches, keeping the `===` header lines in the log so each cell's cache is identifiable.
  The corrected probe is reproducible now, so this is a real experiment rather than a coin flip.
- **`P2-28` — the probe's default `lr=3e-3` is in a bad spot.** Until it is retuned and documented,
  no `check_learn` PASS/FAIL should be used to close a card, and every existing card closed on one is
  suspect. Also decouple `--image-size` from the model constructor argument (AlexNet is 30.0 M params
  at 160 px vs 58.3 M at 227 px), or document the coupling.
- **`P2-29` — E023's "8/8 PASS" came from the buggy probe.** Probably still true, but it is the
  evidence for `P2-21` and should be re-run with the fixed probe.
- **`P3-7`** (paper-faithful `Resize(256)`→`RandomCrop(224)` with padding) remains open. E029's
  "160 px AlexNet train arm fails" support for it is **not** reliable — with the fixed probe, 23/24
  cells pass at 160 px — so treat `P3-7` as a fidelity improvement, not a bug fix.

Still open from before, unchanged: `P2-12`/`P1-5` (Dockerfile), `P2-9` (no tests/lint/CI), `P2-16`,
`P2-17`, and the `P1-*` cleanups.

Known and accepted gaps in *this* session's work:

- The converged runs were stopped early, so there is no final checkpoint accuracy and no SWA result
  (SWA starts at 80 % of `max_epochs`, which neither run reached).
- `lr` was never swept. `3e-4` for AlexNet is a safe Adam default, not a tuned value.
- Only `cuda` was swept for the train arm; CPU was left alone because it is 12–18× slower for
  the same verdict.
- `ColorJitter` and `RandomHorizontalFlip` were not ablated individually.
- **The `lr` sweep is incomplete.** `lr=1e-3` is confirmed at seeds 42 and 3 (16/16 PASS) and
  `lr=1e-2` at the same two seeds (2/16); seeds 1 and 2 were never run at either `lr`, and there is
  no `lr`×`--steps` grid. The sweep was still running when this session was cut off — the container
  `ppg-dev5` and its `/tmp/ppglogs` were removed, so re-run from the commands in E032.
- The seed sweep at 227 px was captured with a `grep` that dropped the `===` header lines, so those
  cells' cache is not recorded in the log. Re-run without the grep if it matters.

---

## 9. Exactly what the next agent should do

1. **Read** this file, [KANBAN.md](../KANBAN.md) and [goal.md](../goal.md). Do not re-read
   HANDOFF_001–004 unless you need a specific decision.
2. **Verify the environment** with `python probes/check_hardware.py` and expect `GPU_OK`. If it is
   not `GPU_OK`, stop and record the verdict — do not train on CPU and draw conclusions about
   convergence from it.
3. **Run `P2-25` first.** It is one command, ~20 minutes to a verdict, and it is the only thing
   standing between this repo and a defensible statement about which optimiser AlexNet should ship
   with. Write the evidence file, then decide `train_final.py`'s `optimizer` accordingly.
4. **Then `P2-27`** — it is an XS/S card that makes every future number mean something. Re-run one
   converged configuration afterwards so the new `val_samples` has a baseline.
5. **Then `P2-15`, at a sane `lr`.** The corrected probe is reproducible, so the augmentation A/B is
   finally runnable as a real experiment. Do it at `--lr 1e-3` (or whatever `P2-28` settles on) and
   ≥3 seeds; a single run at the default `lr` will only reproduce noise.
6. **Do not re-run the E030/E031 configurations to "confirm" them.** They are reproducible from the
   commands in the evidence files; if a number does not reproduce, that is a bug report against the
   evidence file. Note that those runs do **not** depend on `check_learn.py` — the probe bugs in
   §3.7–§3.9 do not touch them.
7. **Keep the probe/deliverable distinction straight.** Probes are cheap and should stop as soon as
   the question is answered — use `--max_epochs` and `--limit_train_batches`, and rely on the
   `EarlyStopping` that is already in both trainers. The 60/100-epoch runs belong to the human on
   the bigger GPU. If you find yourself launching a 60-epoch probe, ask what assumption it tests.
8. **`train_final.py` is the deliverable.** Any number you bake into a `CONFIG` must name the run it
   came from in a comment, and any number you did *not* measure must say so. Re-run both `--smoke`
   paths after touching either file, and validate the dict against `train.py`'s argparse — a key
   argparse does not know about is a crash on the human's machine, not a warning here.
9. **Before closing a card on a probe, re-run the probe's own documented command a second time in a
   fresh process.** That single habit is what invalidated E029 in this session: the first run's
   verdict was not reproducible, and it took one repeat to show the harness, not the network, was
   deciding. If a probe has an A/B in it, also confirm both arms receive the *same object*.

---

## 10. PR

- **Branch:** `agent/convergence-and-augmentation` → `main`
- **Title:** `Land the first converged runs; invalidate the augmentation A/B (check_learn had two RNG bugs)`
- **Cards closed:** `P2-14`, `P2-11` · **reopened:** `P2-15` · **advanced:** `P3-13` · **new:** `P2-25`, `P2-26`, `P2-27`, `P2-28`, `P2-29`
- **Files:** `probes/check_learn.py` (train arm + two RNG seeding fixes + median/max),
  `AlexNet/train.py` (`--gradient_clip_val`), `AlexNet/train_final.py` and
  `GoogLeNet/train_final.py` (measured `CONFIG`), `repos_log/AGENT_INSTRUCTIONS.md`
  (`--entrypoint bash`), `repos_log/goal.md` (Current status only), `evidences/E029` (SUPERSEDED),
  `E030`, `E031`, `E032`, `evidences/INDEX.md`, `KANBAN.md`, `HANDOFF_005.md`
- **Not merged.** Per `AGENT_INSTRUCTIONS.md` §2, the human merges.
