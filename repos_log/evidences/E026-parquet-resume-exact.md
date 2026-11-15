# E026 — Parquet download resume is byte-exact

| | |
|---|---|
| **Status** | CONFIRMED — `VERDICT: RESUME IS EXACT` |
| **Cards** | `P3-11` |
| **Command** | `python probes/check_resume.py --train-samples 8192 --val-samples 512 --seed 7 --kill-after 51` |
| **Host** | `ppg-dev` |
| **First recorded** | [HANDOFF_004 §6.7](../handoffs/HANDOFF_004.md) |

## Method

Two phases: an uninterrupted reference download, then the same cache key built again with the
process killed mid-download and allowed to resume. Labels and paths are fingerprinted and compared.

## Output

```
=== phase 1: uninterrupted reference run ===
reference finished in 89s
reference fingerprint: 8192 labels, first5=[142, 968, 388, 30, 905]

=== phase 2: kill mid-download, then resume ===
  [killed after 51s]
partial parts in ILSVRC__imagenet-1k_8192t_512v_auto_72fd332dd564/_train_parts:
    2 files, manifest=yes ['part-0000.parquet', 'part-0001.parquet']

--- resuming ---
  train: resuming after 6 row groups (5229 samples, 2 part files)
resume finished in 42s (total 93s)
RESUME EVIDENCE: train: resuming after 6 row groups (5229 samples, 2 part files)

labels identical to uninterrupted run: True
paths  identical to uninterrupted run: True

VERDICT: RESUME IS EXACT -- the interrupted download resumed from its checkpoint and
reproduced the uninterrupted subset byte for byte.
```

## The `_manifest.json` trap

The first version of this probe reported `FAIL: parquet parts without a manifest` **while
simultaneously** reporting `labels identical: True` / `paths identical: True` and a clean
`resuming after 6 row groups` line. The contradiction was the probe's own: the file is
**`_manifest.json`** (leading underscore), `utils/datasets_utils.py:201`. Without it the parts are
deliberately deleted as untrusted, so its presence is precisely what makes resume possible.
Checking for `manifest.json` yields a false FAIL.

## Not to be confused with

This is **download** resume. Checkpoint/optimizer resume is
[E017](./E017-resume-roundtrip-on-load-checkpoint-bug.md).
