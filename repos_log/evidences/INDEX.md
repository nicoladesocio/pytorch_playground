# Evidence index — pytorch_playground

Every evidence file here is **self-contained**: it holds the command, the host, the raw output,
and the interpretation. If a kanban card points here, read only that file — you do **not** need to
open the handoff it came from. Handoffs are for narrative and decisions; this folder is for proof.

## Contract

- One evidence piece = one file. Never merge two experiments into one file.
- ID is `E` + three digits, **append-only**. Never renumber, never reuse a retired ID. New
  evidence takes the next free number.
- Filename: `E<NNN>-<kebab-case-summary>.md`.
- Every file starts with the same metadata table: **Status**, **Cards**, **Command**, **Host**,
  **First recorded**. The `Command` must be copy-pasteable — if you cannot reproduce it, it is an
  anecdote, not evidence.
- Raw output goes in fenced blocks **verbatim**. Interpretation goes below the block, never inside
  it. If you correct a number, add a correction note; do not silently edit history.
- A card moves to `DONE` only when its evidence file exists here. A `DONE` row whose evidence link
  is broken is not done.
- Superseded evidence stays, marked `SUPERSEDED` in **Status**, with a pointer to what replaced it
  and a note on why the old claim was wrong. Deleting it destroys the record of how the repo was
  misled — see `E004` and `E005`.

## Index

| ID | Title | Status | Cards |
|---|---|---|---|
| [E001](./E001-trainer-cli-invocation-broken.md) | The documented `trainer` CLI invocation is broken | REPRODUCED (open) | `P1-3` |
| [E002](./E002-set-max-epochs-cannot-work.md) | `--set max_epochs=…` cannot work | REPRODUCED → fixed | `P2-3` |
| [E003](./E003-no-seeding-anywhere.md) | No seeding anywhere in the repo | REPRODUCED → fixed | `P2-2` |
| [E004](./E004-host-cannot-train-environment.md) | Host had no usable training environment | SUPERSEDED (host-specific) | env / `P2-19` |
| [E005](./E005-gpu-blocked-on-one-host.md) | GPU blocked on the HANDOFF_002 host | SUPERSEDED (host-specific) | `P0-5`, `P2-19` |
| [E006](./E006-cpu-fit-works-both-models.md) | Both models complete a real `Trainer.fit()` on CPU | CONFIRMED | `P0-5`, `P2-19` |
| [E007](./E007-image-drift-repo-dockerfile.md) | Image in use was not built from the repo `Dockerfile` | CONFIRMED (open) | `P2-12`, `P1-5`, `P2-20` |
| [E008](./E008-pip-install-e-dot-missing-trainer.md) | `pip install -e .` does not install the `trainer` CLI | REPRODUCED → fixed | `P1-7` |
| [E009](./E009-alexnet-flatten-size-latent-crash.md) | AlexNet's hardcoded `256*6*6` is a latent crash | REPRODUCED → fixed | `P2-13` |
| [E010](./E010-lightning-import-claim-corrected.md) | HANDOFF_001's `pytorch_lightning` premise was wrong | CORRECTED | `P1-2` |
| [E011](./E011-subset-download-parquet-path.md) | Parquet row-group download works at flat memory | CONFIRMED | data path, `P3-14` |
| [E012](./E012-canonical-subset-statistics.md) | Canonical 51200/2048 subset statistics + cache-key/seed gotcha | CONFIRMED | `P1-8`, `P3-14` |
| [E013](./E013-accuracy-metric-aggregation.md) | Accuracy metric: measured Lightning aggregation | CONFIRMED → fixed | `P2-1` |
| [E014](./E014-overfit-one-batch-cpu.md) | Overfit-one-batch on CPU: both architectures learn | CONFIRMED | `P2-11`, `P2-17` |
| [E015](./E015-aux-heads-collapse-small-images.md) | GoogLeNet's aux heads collapse below ~128 px | CONFIRMED | `P2-16` |
| [E016](./E016-kanban-acceptance-gate-40-of-40.md) | Kanban acceptance gate: 40/40 mechanical checks | CONFIRMED | HANDOFF_003 batch |
| [E017](./E017-resume-roundtrip-on-load-checkpoint-bug.md) | Checkpoint resume round-trip + `on_load_checkpoint` bug | CONFIRMED → fixed | `P2-5` |
| [E018](./E018-registry-portability.md) | Registry stores paths relative to the repo root | CONFIRMED → fixed | `P3-5` |
| [E019](./E019-chance-level-per-network.md) | Chance level differs per network: `1.6·ln(K)` | CONFIRMED | `P2-17` |
| [E020](./E020-cpu-smoke-bit-exact-parity.md) | CPU smoke runs reproduce bit-exactly on a 2nd machine | CONFIRMED | `P0-5`, `P2-19` |
| [E021](./E021-gpu-smoke-runs.md) | GPU smoke runs: mixed precision executes on both models | CONFIRMED (executes only) | `P3-13`, `P2-22` |
| [E022](./E022-overfit-one-batch-gpu.md) | Overfit-one-batch on GPU: both architectures learn | CONFIRMED | `P2-11`, `P2-23` |
| [E023](./E023-learning-matrix-both-devices-both-caches.md) | Learning matrix: real images, both devices, both caches (8/8) | CONFIRMED — **unverified**: produced by `check_learn.py` before the two RNG bugs in [E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md) were fixed (`P2-29`) | `P2-21`, `P2-15` |
| [E024](./E024-canonical-subset-correct-cache-key.md) | Canonical subset at the correct `seed=42` cache key | CONFIRMED | `P1-8` |
| [E025](./E025-label-image-alignment-through-sampler.md) | Label/image alignment through the real sampler | CONFIRMED | `P2-18` |
| [E026](./E026-parquet-resume-exact.md) | Parquet download resume is byte-exact | CONFIRMED | `P3-11` |
| [E027](./E027-hardware-probe-gpu-ok.md) | Hardware probe verdict: `GPU_OK` | CONFIRMED | `P2-19` |
| [E028](./E028-argument-contract-rejects-unknown-args.md) | `train_final.py` rejects unknown arguments | CONFIRMED → fixed | `P2-24` |
| [E029](./E029-augmentation-ab-both-arms-identical-images.md) | Augmentation A/B: both arms on identical images; train arm memorises at 227 px | **SUPERSEDED** by E032 — the two arms did not share a batch and the batch was unseeded | `P2-15` (reopened), `P2-21` |
| [E030](./E030-googlenet-converged-run.md) | GoogLeNet converges: `val_acc` 34.6 % vs 1.56 % chance | CONFIRMED | `P2-14`, `P2-11`, `P2-10` |
| [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md) | AlexNet: Adam converges, SGD pinned at chance by `gradient_clip_val=1.0` | Measurements CONFIRMED; **causal claim superseded** by E033 | `P2-14`, `P2-11`, `P2-12`, `P2-25` |
| [E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md) | `check_learn.py` had two RNG bugs; its verdict is dominated by its own default `lr` | CONFIRMED | `P2-15`, `P2-28`, `P2-29`, `P2-21` |
| [E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md) | AlexNet SGD: 12 arms across clip × lr × precision, plus two 7 680-step arms, all on chance; Adam control at the same budget leaves chance | CONFIRMED (negative) — **supersedes the causal claim in E031**; `P2-25` stays open | `P2-25`, `P2-12` |
| [E034](./E034-alexnet-init-gradient-norms-span-1e4-googlenet-does-not.md) | AlexNet's per-tensor init gradient norms span 1.5 × 10⁴; GoogLeNet's span 6.3 × 10² | CONFIRMED | `P2-25`, `P2-12` |
| [E035](./E035-alexnet-init-is-not-what-blocks-sgd.md) | The uniform `std=0.01` init is not what blocks SGD (memorises 3/4 fixed batches); `kaiming` is actively broken | CONFIRMED (negative) | `P2-25`, `P2-12` |
| [E036](./E036-validation-pool-size-and-accuracy-resolution.md) | Validation is 153 images (0.654 %); `val_samples=16384` gives 1030 (0.0971 %) | CONFIRMED — `P2-27` resolved | `P2-27` |
| [E037](./E037-lr-schedule-is-stepped-per-epoch-and-healthy.md) | The cosine schedule is stepped per **epoch** and the LR stays healthy — a source-reading theory, refuted by measurement | CONFIRMED (negative) | `P2-25` |
| [E038](./E038-per-group-lr-sgd-sweep.md) | Per-group feature/head `lr` up to 100× does not rescue SGD (feature displacement saturates); neither do dropout off, `weight_decay=0`, or both | CONFIRMED (negative) | `P2-25` |
| [E039](./E039-e023-learning-matrix-not-reproducible-across-seeds.md) | E023's "8/8 PASS" is not reproducible: 34/36 cells across 3 seeds, and the 2 failures move between caches and devices | CONFIRMED — **supersedes E023**; `P2-29` resolved | `P2-29`, `P2-21`, `P2-28` |

## Reading shortcuts

- **"Can these networks learn?"** → [E030](./E030-googlenet-converged-run.md) and
  [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md) — real training runs, both above
  chance. [E023](./E023-learning-matrix-both-devices-both-caches.md) (real images, memorisation) and
  [E014](./E014-overfit-one-batch-cpu.md) / [E022](./E022-overfit-one-batch-gpu.md) (backprop only)
  are the weaker, earlier forms of the same answer.
- **"Why did earlier runs look flat?"** → [E030](./E030-googlenet-converged-run.md): an epoch on the
  canonical cache is 4 153 samples = 65 steps at batch 64, so 18 "epochs" was ~1 170 optimiser
  steps. `--train_samples_per_epoch` is the knob.
- **"Is augmentation stopping learning?"** → unresolved. [E029](./E029-augmentation-ab-both-arms-identical-images.md)
  said no, but it is **superseded**: its two arms were fed different batches
  ([E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md)). `P2-15` is back in TODO.
- **"Can I trust a `check_learn.py` PASS/FAIL?"** → read
  [E032](./E032-check-learn-probe-bugs-and-lr-sensitivity.md) first. The probe's default `lr=3e-3`
  sits in a bad spot: the same batch/seed flips PASS→FAIL on `lr` alone (`1e-3` → 16/16 PASS,
  `3e-3` → 4/8, `1e-2` → 3/16). Never close a card on one unseeded `check_learn` run.
- **"Why is AlexNet's SGD arm at chance?"** → **not** the clip. [E031](./E031-alexnet-adam-converges-sgd-pinned-at-chance.md)
  said `gradient_clip_val=1.0` scales every step by 0.037; that causal claim is **superseded** —
  clip off and clip=30 are also exactly at chance
  ([E033](./E033-alexnet-sgd-grid-clip-init-and-budget-ruled-out.md)). **Eight** candidate causes
  have since been ruled out by measurement: the clip and fp16 (E033), the step budget (E033's Adam
  control and two 7 680-step arms), the `std=0.01` init
  ([E035](./E035-alexnet-init-is-not-what-blocks-sgd.md)), the cosine schedule
  ([E037](./E037-lr-schedule-is-stepped-per-epoch-and-healthy.md)), the 1.5 × 10⁴ gradient spread,
  dropout and `weight_decay`
  ([E038](./E038-per-group-lr-sgd-sweep.md)). The surviving suspect is the data: only 4 153 unique
  in-class train images, and `train_acc` never exceeds 1.95 % in any SGD arm. `P2-25` is still open;
  Adam remains the shipped optimiser, and a ninth optimiser-side variant is not worth running.
- **"What `val_samples` should I use?"** → 16384. At the old 2048 the 64 chosen classes hold only
  153 validation images, so every `val_acc` in the repo is a multiple of 0.654 %
  ([E036](./E036-validation-pool-size-and-accuracy-resolution.md)). Note the pool size also changes
  *which* 64 classes are chosen, so accuracy figures are not comparable across pools.
- **"Before 'fixing' a schedule or an init from reading the source"** →
  [E037](./E037-lr-schedule-is-stepped-per-epoch-and-healthy.md). A `CosineAnnealingLR(T_max=max_epochs)`
  returned as a bare dict *looks* like a steps-vs-epochs units bug and is not one; and
  [E035](./E035-alexnet-init-is-not-what-blocks-sgd.md) shows `kaiming_normal_(mode='fan_in')` makes
  AlexNet's init **worse**, not better. Measure first.
- **"Is the GPU usable?"** → run `python probes/check_hardware.py`; see
  [E027](./E027-hardware-probe-gpu-ok.md) for what a passing verdict looks like. Do **not** read
  [E004](./E004-host-cannot-train-environment.md) or
  [E005](./E005-gpu-blocked-on-one-host.md) as rules — they are one machine's measurements.
- **"Before writing a probe"** → [E025](./E025-label-image-alignment-through-sampler.md) (a
  `WeightedRandomSampler` redraws its order per iteration; a probe that can only fail one way is
  not a probe) and [E016](./E016-kanban-acceptance-gate-40-of-40.md) (a gate that only ever passes
  is not a gate).
- **"What accuracy numbers can I trust?"** → anything before HANDOFF_003, no. See
  [E013](./E013-accuracy-metric-aggregation.md).
