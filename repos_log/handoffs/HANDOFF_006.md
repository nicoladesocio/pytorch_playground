# HANDOFF_006 — P2-27 closed with a measured number; P2-25's clip theory killed by measurement

**Agent:** sixth agent (session 6)
**Date:** 2026-11-16
**Host:** `LX0350043`, native Ubuntu 22.04.5, i7-13800H (20 cores), 62 GB RAM
**GPU:** NVIDIA RTX A1000 6GB Laptop GPU — Ampere `sm_86`, 6144 MiB, driver 535.309.01, CUDA 12.2
**Branch:** `agent/sgd-clip-verdict-and-val-pool` (branched from `main` at `a12117e`)
**Cards:** `P2-27` **DONE** · `P2-25` **still open** (eight causes ruled out by measurement) · `P2-15`/`P2-28`/`P2-29` **not started**
**Read next:** only this file + [KANBAN.md](../KANBAN.md) + [goal.md](../goal.md).

> Written incrementally, per `AGENT_INSTRUCTIONS.md` §2.3.

---

## 0. TL;DR for the next agent

1. **`P2-27` is closed with a measured number, not a guess.** The shipped validation pool was
   **153** in-class images → accuracy resolution **0.6536 %**, so the "1.56 % chance level" quoted
   across the repo was 24 images, and every difference under 0.65 % was noise. `val_samples=16384`
   gives **1030** images → **0.0971 %**, a 6.7x improvement, and it is now the shipped value in both
   `train_final.py` files. Evidence: [E036](../evidences/E036-validation-pool-size-and-accuracy-resolution.md).
2. **`P2-25` is NOT closed, and HANDOFF_005 §0.3's explanation for it was wrong.** The claim
   "AlexNet's SGD arm is pinned at chance **by `gradient_clip_val=1.0`**" is refuted: clip off and
   clip 30 are both exactly at chance. So are `bf16-mixed`, a 3.8x longer budget, three inits,
   per-group learning rates up to 100x, dropout removed entirely, and `weight_decay=0`. **Eight
   causes, all eliminated by measurement** — see §3.1. Do not spend a ninth on another optimiser
   or regulariser variant.
3. **The strongest surviving clue is the data, not the optimiser.** In every SGD arm `train_acc`
   never exceeds **1.95 %** — SGD fails to fit even its own training stream — and pushing the
   feature-extractor learning rate 100x raises weight displacement only **1.56x** (saturating).
   The features are not short of learning rate, they are short of signal. The data is now the
   **only** surviving suspect.
4. **`val_samples` is not a neutral knob — this is the caveat E036 did not anticipate.** It feeds
   `choose_target_classes`'s `min_val_count` filter, so 2048 vs 16384 selects **8 different classes
   out of 64** and moves the in-class train count 4153 → 4195. Numbers from the two pools are not
   comparable, and the E030/E031 figures are therefore not reproducible under the new default.
5. **Two of my own hypotheses died to measurement, and both are recorded so you don't repeat them.**
   The cosine schedule is *not* a steps-vs-epochs units bug (E037), and the `std=0.01` init is *not*
   what blocks SGD (E035). Both looked obviously true from reading the source.

---

## 1. Why this session exists

HANDOFF_005 §9 gave me, in priority order: `P2-25` (run the unclipped-SGD experiment that would
either close or kill the clip theory), `P2-27` (decide `val_samples` from a measurement), then
`P2-15`/`P2-28`/`P2-29`.

I delivered `P2-27` completely and drove `P2-25` to a defensible verdict — but the verdict is
"still open, and here are six things it is not". I did not start `P2-15`/`P2-28`/`P2-29`; §9 tells
you exactly what they need.

---

## 2. Step log

| # | Command | Result |
|---|---|---|
| 1 | `docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_hardware.py'` | `PASS=7 FAIL=0 WARN=0`, `VERDICT: GPU_OK` — verbatim in §3.0 |
| 2 | `probes/check_lr_schedule.py` (new) | LR is a clean two-level staircase `1.0e-2 → 5.0e-3`, 0/600 steps below 1 % of nominal. **My scheduler theory was wrong.** E037 |
| 3 | `train.py --optimizer sgd --lr 0.1 --gradient_clip_val 30 --train_samples_per_epoch 512 --max_epochs 15` | `val_loss 4.170`, chance. Clip 30 does not help |
| 4 | same with `--gradient_clip_val 0 --lr 1e-2 --train_samples_per_epoch 512 --max_epochs 15` | `val_loss 4.170`, chance. 7 680 steps of unclipped SGD |
| 5 | `train.py --optimizer adam ...` (control, re-run in a fresh process) | `4.120 → 3.850`, `0.654 → 7.190 %` — reproduces E031's numbers **to the digit** |
| 6 | `probes/check_sgd_vs_init.py --batches 4` (after fixing it) | paper **3/4**, torch **4/4**, kaiming **0/4**. **My init theory was wrong.** E035 |
| 7 | `probes/check_grad_scale.py` (re-run) | reproduces exactly: total 26.7148, spread 1.518e+04, ratio 5.317 |
| 8 | `probes/check_sgd_lr_groups.py --ratio 1,3,10,30,100` (new) | all chance; `feat \|dw\|` saturates at 1.960e+01. E038 |
| 9 | `probes/check_sgd_lr_groups.py --ratio 1 --dropout off` | `val_loss 4.1741`, chance. Dropout is not it either |
| 10 | `probes/check_val_pool.py --val_samples 2048,16384,32768 --predict` (new) | 153 / 1030 / 2054 images; 0.6536 % / 0.0971 % / 0.0487 %. E036 |
| 11 | `probes/check_sgd_lr_groups.py --ratio 1 --weight_decay 0` (and the `--dropout off` twin, plus a fresh-process replicate) | all `val_loss 4.1748`, chance. `weight_decay` is not it either; replicate reproduced to the digit |

---

## 3. Findings

### 3.0 The mandatory probe, verbatim

```
$ docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_hardware.py'
========================================================================
pytorch_playground hardware probe
========================================================================

--- host / driver ---
[INFO] platform — linux
[PASS] nvidia-smi — NVIDIA RTX A1000 6GB Laptop GPU, driver 535.309.01, 6144 MiB
[PASS] driver version — 535.309.01 supports the CUDA 12.x runtime

--- container runtime ---
[INFO] container runtime — running inside a container — host runtime checks skipped (run `python probes/check_hardware.py --host-only` on the host to see them)

--- shared memory (DataLoader workers) ---
[PASS] /dev/shm — 8.6 GB
[INFO] CPUs visible — 20

--- torch ---
[INFO] torch — 2.10.0+cu128 (CUDA built 12.8)
[INFO] torchvision — 0.25.0+cu128
[INFO] lightning — 2.6.6
[PASS] torch.cuda — 1 device(s): NVIDIA RTX A1000 6GB Laptop GPU, sm_86, 6.2 GB
[PASS] wheel supports device arch — sm_86 in ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
[INFO] mixed precision — bf16=True (needs sm_80+); fp16 available on all CUDA archs
[PASS] AlexNet gpu fwd/bwd — loss=3.1570 grads=16/16 0.43s (fp32, B=4, 160px)
[PASS] GoogLeNet gpu fwd/bwd — loss=3.6819 grads=126/126 0.20s (fp32, B=4, 160px)

========================================================================
PASS=7  FAIL=0  WARN=0
VERDICT: GPU_OK
         CUDA forward+backward verified on both networks
         NVIDIA RTX A1000 6GB Laptop GPU (sm_86, 6.2 GB)
         => Run the smoke tests on GPU. Do NOT fall back to CPU and do NOT
            cite AGENT_INSTRUCTIONS.md's old 'GPU is unusable' note.
========================================================================
```

The `--host-only` half of HANDOFF_005 §9 step 1 was **not** run: this session is container-only and
the host has no `python` on `PATH` (`/bin/bash: line 1: python: command not found`). The in-container
run is the one that gates the work, and it passes.

### 3.1 P2-25: eight causes eliminated, and the clip theory is dead

Nineteen SGD configurations across E033 and E038. Every one lands on chance
(`val_loss 4.1589 = ln 64`):

| Candidate | Verdict | Evidence |
|---|---|---|
| `gradient_clip_val=1.0` | **eliminated** — off and 30 both at chance | E033 |
| fp16 overflow | **eliminated** — `bf16-mixed` at chance | E033 |
| Step budget too short | **eliminated** — Adam control converges at the same budget; 7 680-step SGD arms flat | E033 |
| Uniform `std=0.01` init | **eliminated** — memorises 3/4 fixed batches | E035 |
| Cosine schedule | **eliminated** — stepped per epoch, LR healthy | E037 |
| 1.5 × 10⁴ gradient spread | **eliminated** — per-group `lr` up to 100× does nothing | E038 |
| `dropout p=0.5` | **eliminated** — all dropouts → `nn.Identity`, still chance | E038 |
| `weight_decay=5e-4` | **eliminated** — `wd=0` at chance, alone and combined with dropout off | E038 |
| 4 153 unique training images | **untested**, and now the *only* surviving suspect | — |

Two measurements are worth more than the table. First, `train_acc` never exceeds **1.95 %** in any
SGD arm — SGD is not "fitting but failing to generalise", it is failing to fit. Second, `feat |dw|`
saturates: ratio 30 → 100 (a 3.3x LR increase) moves the features 1.03x. Whatever is limiting the
feature extractor is not the learning rate.

Also: `kaiming_normal_(mode='fan_in', nonlinearity='relu')` is **actively broken** on this net —
first loss 8.5–9.9 on a 10-class task (chance 2.3026), never recovering. Do not "modernise"
`AlexNet/model.py`'s init without re-running `check_sgd_vs_init.py` first.

### 3.2 P2-27: the number, and the caveat

| `val_samples` | in-class val images | resolution | classes differ from 2048 |
|---|---|---|---|
| 2048 (old default) | **153** | 0.6536 % | — |
| **16384 (shipped)** | **1030** | **0.0971 %** | 8 of 64 |
| 32768 | 2054 | 0.0487 % | more |

16384 is the shipped value: 6.7x better resolution for a 7.6 GB cache and ~4 s of validation per
epoch. 32768 is available but changes the class set further and doubles the cache.

The caveat: `val_samples` is an input to `choose_target_classes`'s `min_val_count` filter. Changing
it changes **which 64 classes the task is** — 8 differ between 2048 and 16384, and the in-class
train count moves 4153 → 4195. So the E030/E031 accuracy figures cannot be reproduced under the new
default, and E033 deliberately pinned `--val_samples 2048` to stay comparable with E031.

### 3.3 Two of my own theories were wrong

I wrote both up as evidence rather than quietly dropping them, because they are the two theories
most likely to occur to the next agent from reading the source alone.

- **The scheduler.** `CosineAnnealingLR(T_max=max_epochs)` returned as a bare dict *looks* like a
  steps-vs-epochs units bug that would collapse LR to zero in the first few batches. It is not one.
  Lightning steps it per **epoch**; the measured trajectory is a clean two-level staircase. Had I
  "fixed" this I would have broken a working component. → E037
- **The init.** "Uniform `std=0.01` gives degenerate gradients, that must be it" survived exactly
  one run. The original single-batch probe said the paper init memorised; on a fresh process it
  didn't. The bug was the probe: one 32-image batch is too thin to decide the question, and
  `find_caches()` sorts by mtime so the cache silently followed whichever was built last. Fixed with
  `--batches` (default 4) and a pinned `--cache` default. → E035

---

## 4. Code changes

| File | Change |
|---|---|
| `probes/check_lr_schedule.py` | **new** — samples `param_groups[*]['lr']` in `on_before_optimizer_step` during a real `Trainer.fit()`; verdicts the schedule |
| `probes/check_val_pool.py` | **new** — counts in-class val images per pool size, reports accuracy resolution, `--predict` extrapolates |
| `probes/check_sgd_lr_groups.py` | **new** — feature/head per-group LR sweep with `max\|dw\|` displacement, plus `--dropout off` |
| `probes/check_sgd_vs_init.py` | **fixed** — `--batches` (default 4) with per-batch re-seeding, `--cache` pinned to `51200t_2048v`, summary rewritten to "memorised N/M" |
| `AlexNet/train_final.py` | `CONFIG["val_samples"]` 2048 → **16384** with the resolution + class-drift comment; optimizer comment rewritten to cite E033–E038 instead of the refuted clip theory |
| `GoogLeNet/train_final.py` | `CONFIG["val_samples"]` 2048 → **16384** |
| `evidences/E033`–`E038` | **new** — six files, one per experiment |
| `evidences/E031-...md` | Status → **PARTIALLY SUPERSEDED** (measurements stand, causal claim refuted) |
| `evidences/INDEX.md` | six new rows, E031 row corrected, two new reading shortcuts |
| `goal.md` | AlexNet "Trainable?" row corrected; P2-27 paragraph added |
| `KANBAN.md` | `P2-27` → DONE; `P2-25` body rewritten |

---

## 5. How to reproduce

```bash
# 0. container (note --entrypoint bash, per HANDOFF_005 §7)
docker run -d --name ppg-dev6 --gpus all --shm-size=8g \
  -v "$PWD:/workspace" --entrypoint bash ppg-gpu:latest -c 'sleep infinity'

# 1. gate
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_hardware.py'

# 2. P2-27 (E036) — ~2 min, no training
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_val_pool.py \
  --val_samples 2048,16384,32768 --predict'

# 3. P2-25 grid (E033) — ~12 min per arm; keep to two concurrent
docker exec ppg-dev6 bash -lc 'cd /workspace && python train.py --network alexnet \
  --optimizer sgd --lr 0.1 --gradient_clip_val 0 --precision 16-mixed \
  --train_samples_per_epoch 512 --max_epochs 15 --val_samples 2048'

# 4. P2-25 init (E035) — ~4 min; --batches matters, see §3.3
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_vs_init.py --batches 4'

# 5. P2-25 per-group LR + dropout (E038) — ~6 min/ratio
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1,3,10,30,100'
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1 --dropout off'

# 5b. P2-25 regularisers (E038, Command C) — ~3 min each. NB the probe exits non-zero
#     when no arm escapes chance, so chain with ';' not '&&'.
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1 --weight_decay 0'
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_sgd_lr_groups.py \
  --steps 256 --epochs 8 --lr 0.1 --ratio 1 --weight_decay 0 --dropout off'

# 6. P2-25 schedule (E037) — ~3 min
docker exec ppg-dev6 bash -lc 'cd /workspace && python probes/check_lr_schedule.py'
```

---

## 6. Evidence

| File | One-line summary |
|---|---|
| [E033](../evidences/E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md) | 12-arm SGD grid: clip, precision and budget all ruled out; Adam control reproduces to the digit |
| [E034](../evidences/E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md) | init gradient norms span 1.5 × 10⁴ on AlexNet, 1.2 × 10² on GoogLeNet |
| [E035](../evidences/E035-alexnet-init-is-not-what-blocks-sgd.md) | paper init memorises 3/4 fixed batches — the init is not the blocker; documents the probe fix |
| [E036](../evidences/E036-validation-pool-size-and-accuracy-resolution.md) | P2-27's number: 153 → 1030 images, 0.6536 % → 0.0971 %, plus the class-drift caveat |
| [E037](../evidences/E037-lr-schedule-is-stepped-per-epoch-and-healthy.md) | the cosine schedule is healthy — written as an explicit refutation of my own theory |
| [E038](../evidences/E038-per-group-lr-sgd-sweep.md) | per-group LR up to 100× and dropout off do nothing; `feat \|dw\|` saturates |

---

## 7. Environment notes

- **`docker exec` needs the container started with `--entrypoint bash`** (HANDOFF_005 §7). Still true.
- **`python -u` for backgrounded runs.** `nohup python` without `-u` buffers everything, so progress
  polls show nothing for minutes and you cannot tell a hung job from a slow one.
- **Two concurrent `16-mixed` 227 px runs fit in 6 GB (~3.2 GB each); a third OOMs.**
- **`find_caches()` sorts by mtime.** Any probe whose `--cache` filter matches multiple caches
  silently follows whichever was built last. This is what made `check_sgd_vs_init.py`
  non-reproducible; its default is now pinned. If you add a probe that picks a cache, pin it too.
- **`_cache_key` includes `val_samples` but not `shuffle_buffer_size`.** The 16384 cache is a
  separate 7.6 GB directory. Disk was at 90 % (95 G free) — check before building another pool.
- **A NaN run still exits 0 and still writes `alexnet_final.ckpt`** (EarlyStopping's finite guard +
  `divergence_threshold=50.0`). Grep the log; never trust the exit code.
- **Lightning's `\r` progress bar** needs `tr -d '\r' | grep -aoE 'val_loss=[0-9.nan]+, val_acc=[0-9.]+, ...' | awk '!seen[$0]++'`, and each epoch appears twice.
- **`val_acc` is quantised.** With 153 images the possible values are multiples of 0.6536 %; the raw
  logs show 0.654, 1.310, 1.960, 2.610, 5.230, 7.190. Differences of one or two images are not signal.
- **Auth:** the injected `GH_TOKEN` is invalid (401). Push with
  `env -u GH_TOKEN -u GITHUB_TOKEN git push origin HEAD:<branch>`; `gh` uses the keyring identity
  `nicoladesocio`. SSH is denied.

---

## 8. Open

| Card | State |
|---|---|
| `P2-25` | **open** — eight causes eliminated; only the 4 153-image data limit is untested |
| `P2-15` | **not started** — augmentation A/B, needs ≥3 seeds and both caches |
| `P2-28` | **not started** — retune `check_learn.py`'s default `lr`/`--steps` |
| `P2-29` | **not started** — re-confirm or supersede E023 |
| `P3-13` | still IN PROGRESS from agent 5 |

---

## 9. Exactly what the next agent should do

1. **Verify the environment** with `python probes/check_hardware.py` and expect `GPU_OK`.
2. **`P2-25` — do not write another optimiser *or* regulariser variant.** Eight are measured, and
   `weight_decay=0` is now among them (`val_loss 4.1748`, chance, reproduced in a fresh process).
   The one experiment left is the data:
   - **Raise the effective training set well above 4 153 unique images**, or relax
     `min_val_count`/class selection so more images are available, and see whether `train_acc` moves
     off 1.95 %. Note that `--train_samples_per_epoch 16384` is already what the converged Adam runs
     used, and it draws *with replacement* from 4 153 unique images — so more draws per epoch is not
     more data. If SGD still cannot fit its own training stream, the answer is not in the optimiser
     at all and `P2-25` should be re-scoped rather than closed.
   - Sanity-check the framing first: the Adam arm converges on the *same* 4 153 images, so the data
     is not unlearnable in principle. What that leaves is an interaction between SGD and something
     about this data that Adam's per-parameter scaling happens to paper over.
3. **`P2-15`** — `check_learn.py --arm both --lr 1e-3 --image-size 227`, **≥3 seeds**, on **both**
   caches, preserving the `===` headers in the evidence file. HANDOFF_005 §0.2: the previous close
   was invalid, so this is a fresh measurement, not a confirmation.
4. **`P2-28`** — retune `check_learn.py`'s default `lr`/`--steps` from a sweep and put the sweep in
   an evidence-named comment; document the `--image-size` ↔ constructor coupling.
5. **`P2-29`** — re-run `check_learn.py --device both --steps 200` with **≥3 seeds** and either
   re-confirm E023 or mark it SUPERSEDED.
6. **Do not re-run E030/E031 configs to "confirm" them** (HANDOFF_005 §9). They are settled.
7. **Double-run rule (§9.9).** Before closing a card, re-run its documented command a second time in
   a fresh process. E035 is the cautionary tale: the first run was a fluke and only the second
   caught it.

---

## 10. PR

- **Branch:** `agent/sgd-clip-verdict-and-val-pool` → `main`
- **Title:** `P2-27: ship val_samples=16384; P2-25: rule out clip, init, schedule and gradient spread as the cause of AlexNet's chance-level SGD`
- **Cards closed:** `P2-27` · **advanced, not closed:** `P2-25`
- **Files:** `probes/check_lr_schedule.py`, `probes/check_val_pool.py`,
  `probes/check_sgd_lr_groups.py` (new), `probes/check_sgd_vs_init.py` (fixed),
  `AlexNet/train_final.py`, `GoogLeNet/train_final.py`, `evidences/E033`–`E038`,
  `evidences/E031-...md`, `evidences/INDEX.md`, `goal.md`, `KANBAN.md`, `HANDOFF_006.md`
- **PR:** opened — see the URL in the final summary.
- **Not merged.** Per `AGENT_INSTRUCTIONS.md` §2, the human merges.
