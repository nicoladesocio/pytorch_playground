# E011 — Parquet row-group download path works at flat memory

| | |
|---|---|
| **Status** | CONFIRMED |
| **Cards** | data-path unblock (substance of HANDOFF_003), `P3-14` |
| **Command** | `python _check_data.py 4096 512 42` (scratch probe, deleted before the PR) |
| **First recorded** | [HANDOFF_003 §6.1](../handoffs/HANDOFF_003.md) |

## Output

```
Downloading ILSVRC/imagenet-1k subset (4096 train, 512 val samples)...
  train: 4096/4096 samples (1 part files)
  validation: 512/512 samples (0 part files)
Cached for future runs!
Train dataset size: 4096
Validation dataset size: 512
elapsed=208.0s
train: 4096 samples, 986 distinct classes
val:   512 samples, 400 distinct classes
train per-class counts: min=1 p25=3 median=4 p75=5 max=10
chosen 64 classes -> 64
chosen per-class: min=7 median=8 max=10
val samples covering chosen classes: 34 / 512
sample item: <class 'dict'> (96123, 'n02978881_9704_n02978881.JPEG') label 481
```

## What it proves

208 s for 4096+512 samples at **flat RSS**, versus an OOM-kill before the rewrite. The streaming
download-then-train path was structurally wrong for this dataset; downloading parquet row groups
directly is not.

The `34 / 512` line is what motivated requiring validation-set support in `choose_target_classes`
— a class selection that ignores val support leaves you with almost no validation data.

## Related

`save_to_disk` remains the only memory-heavy step in the path; dropping it is `P3-14`.
