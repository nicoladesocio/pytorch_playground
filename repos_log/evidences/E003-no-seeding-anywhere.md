# E003 — No seeding anywhere in the repo

| | |
|---|---|
| **Status** | REPRODUCED → fixed in HANDOFF_003 |
| **Cards** | `P2-2` |
| **Command** | `grep -r "seed_everything\|manual_seed\|random.seed\|np.random.seed"` |
| **First recorded** | [HANDOFF_001 §3c](../handoffs/HANDOFF_001.md) |

## Output

```
grep -r "seed_everything|manual_seed|random.seed|np.random.seed"  →  No matches found
```

## Why it mattered

`create_subset_sampler` called `random.sample(...)` unseeded, so **each run picked a different
64-class subset**. Two consequences: runs were not comparable, and a resumed run whose sampler
re-picked classes could drift from the `target_classes` baked into the checkpoint.

## Resolution

`seed_everything(seed)` in both train scripts, seeded class selection, `--seed` flag
(HANDOFF_003). Reproducibility is now a tested property — see
[E020](./E020-cpu-smoke-bit-exact-parity.md), where the same numbers reproduce on a second,
physically different machine.
