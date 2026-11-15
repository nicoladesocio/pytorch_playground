# E017 — Resume round-trip, and the `on_load_checkpoint` bug it exposed

| | |
|---|---|
| **Status** | CONFIRMED → fixed in HANDOFF_003 |
| **Cards** | `P2-5` |
| **Command** | `python probes/check_resume_rt.py` |
| **First recorded** | [HANDOFF_003 §6.6](../handoffs/HANDOFF_003.md) |

## Method

Trains AlexNet for 2 epochs with the same `ModelCheckpoint` naming `train.py` uses, locates the
checkpoint through `get_latest_checkpoint()`, and re-enters `fit(ckpt_path=...)` with
`max_epochs=4`.

## Output

```
checkpoints: ['alexnet_epoch_epoch=00.ckpt', 'alexnet_epoch_epoch=01.ckpt']
latest -> alexnet_epoch_epoch=01.ckpt
phase3 current_epoch = 4
optimizer state restored = True
scheduler restored = True last_epoch = 4
legacy checkpoint rejected = True
[P2-5] resume round-trip PASS
```

## Two findings

1. **The filename glob is correct, but only by accident of Lightning's defaults.** The callback is
   configured `filename="alexnet_epoch_{epoch:02d}"` and Lightning's `auto_insert_epoch` then
   injects the literal `epoch=`, producing `alexnet_epoch_epoch=00.ckpt` — which is what
   `f"{model_prefix}_epoch=*.ckpt"` matches. If anyone sets `auto_insert_epoch=False` or renames
   the pattern, resume silently returns `None` and the run restarts from scratch. The probe now
   asserts non-`None`, so it fails loudly instead of silently.
2. **`on_load_checkpoint` made current-code checkpoints unresumable.** It tested
   `hyper_parameters.get("target_classes") is None` and raised. But `target_classes=None` is a
   *supported* configuration (output indices then equal the original class labels), so a checkpoint
   written by the current code with that setting was rejected as "pre-remapping". The guard now
   tests `"target_classes" not in hparams` — key **presence**, not value. The probe covers both
   directions: a normal checkpoint resumes, and a checkpoint with the key deleted is still rejected.

## Not to be confused with

This is **checkpoint** resume. Download resume (killing a parquet download mid-flight and
continuing it) is [E026](./E026-parquet-resume-exact.md).
