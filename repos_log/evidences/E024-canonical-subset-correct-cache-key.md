# E024 — Canonical 51200/2048 subset built at the correct `seed=42` cache key

| | |
|---|---|
| **Status** | CONFIRMED |
| **Cards** | `P1-8` |
| **Command** | `python probes/check_big.py` (pinned `SEED = 42`) |
| **Host** | `ppg-dev` container |
| **First recorded** | [HANDOFF_004 §6.5](../handoffs/HANDOFF_004.md) |

## Output

```
549.5 s   peak ~5.8 GB
cache key: ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538
train/  validation/  metadata.json   all present
```

## Why this was a card at all

`train_final.py` bakes `"seed": 42`, and `seed` is part of `_cache_key`. HANDOFF_003 had only ever
built the `seed=1337` key (`..._2ecb016d7b0d`) — a **different directory** — so the deliverable
script could not find its own subset and would start a 13-minute download on every run. See the
gotcha in [E012](./E012-canonical-subset-statistics.md).

## What it unblocks

`train_final.py` now loads its own subset **offline**, with no network and no HF token. That is
what makes the CPU parity check ([E020](./E020-cpu-smoke-bit-exact-parity.md)) and the learning
matrix ([E023](./E023-learning-matrix-both-devices-both-caches.md)) runnable at all.

## Memory note

549.5 s and ~5.8 GB peak. The `save_to_disk` round-trip is the only memory-heavy step and was
OOM-killed at 2/12 shards inside an 8.25 GB container cap on the first attempt — hence `P3-14`.
