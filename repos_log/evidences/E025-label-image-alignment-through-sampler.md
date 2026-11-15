# E025 — Label/image alignment verified through the real sampler

| | |
|---|---|
| **Status** | CONFIRMED — all 3 checks PASS |
| **Cards** | `P2-18` |
| **Command** | `python probes/check_labels2.py` (`--max-samples 400 --image-size 160`) |
| **Host** | `ppg-dev`, canonical cache |
| **First recorded** | [HANDOFF_004 §6.6](../handoffs/HANDOFF_004.md) |

## Why this mattered more than any other probe

If labels did not line up with images, **every accuracy number in the repo would be meaningless**
— including all the convergence evidence. HANDOFF_003 listed this as the highest-stakes untested
assumption.

## Output

```
cache: /workspace/Data/datasets/ILSVRC__imagenet-1k_51200t_2048v_auto_c58e22ff2538
train=51200 val=2048  features=['image', 'label']

[1] label<->wnid bijection over 51200 rows: 1000 labels, 1000 wnids
    labels mapping to >1 wnid: 0
    wnids mapping to >1 label: 0
    OK
    malformed filenames: 0 (OK)

[2] label index vs canonical ImageNet ordering (sorted wnids)
    labels whose wnid is not at its sorted position: 0 (OK)
    extremes: label 0 -> n01440764 (tench, Tinca tinca), label 999 -> n15075141 (toilet tissue, ...)

[3] pipeline check with target_classes=[39, 56, 156, 183, 515, 518, 758, 793, 916, 945]
    inspected 400 sampled rows over 10 remapped indices
    rows whose stored label disagrees with the dataset's remap: 0 (OK)
    remapped indices pointing at >1 wnid: 0 (OK)

RESULT: labels and images agree through the real sampler. Accuracy numbers can be trusted.
```

## The offline ground truth that check 2 adds

The HF repo ships no `class_indices.json` (only `.gitattributes`, `README.md`, `classes.py`), so
there is no file to compare against. But **ImageNet-1k's canonical class index is the 1000 WordNet
ids in lexicographic order** — verifiable offline, and it catches an off-by-one that the
label↔wnid bijection alone cannot see. 0 mismatches over all 1000 labels; label 0 → `n01440764`
"tench", 999 → `n15075141` "toilet tissue".

## Two traps this probe nearly sprang on itself

1. **`WeightedRandomSampler.__iter__` draws a fresh random order on every call.** Calling
   `list(sampler)` and then iterating a `DataLoader` built on the same sampler compares two
   unrelated orders. The first run reported **359/400 mismatches and 10 mixed classes** — a false
   alarm that looked exactly like label corruption. (The 89.75 % rate was almost exactly the
   random rate for 10 classes, which is how it was spotted.) Fixed by freezing the order
   (`sampler=list(sampled)`) before constructing the loader.
2. **Check 3 was originally tautological** — it derived the expected wnid from
   `target_classes[remapped]`, i.e. from the thing under test. It now re-reads the raw row the
   sampler visited and compares its *stored* label against the remapped index the dataset returns.

A guard was added so that a 100 %-mismatch result reports *"this probe is misaligned, not the
cache"* rather than a scary label-corruption message. **A probe that can only fail in one
direction is not a probe.**
