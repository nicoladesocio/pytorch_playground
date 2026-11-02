"""Build the canonical subset used by train_final.py and report its statistics.

Run from anywhere:  python probes/check_big.py
"""
import time
from collections import Counter

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from utils.datasets_utils import (load_dataset_subset, label_distribution,
                                  choose_target_classes, create_subset_sampler)

T, V = 51200, 2048
t0 = time.time()
train, val, _ = load_dataset_subset(train_samples=T, val_samples=V, seed=1337)
print(f"elapsed={time.time()-t0:.1f}s train={len(train)} val={len(val)}")

tl = label_distribution(train, "train")
vl = label_distribution(val, "validation")
print("train classes:", len(set(tl["labels"])))
print("val classes:", len(set(vl["labels"])))
c = Counter(tl["labels"])
q = sorted(c.values())
print("per-class train min/p25/med/p75/max:",
      q[0], q[len(q)//4], q[len(q)//2], q[3*len(q)//4], q[-1])

for K in (10, 64, 128):
    tc = choose_target_classes(train, num_classes=K, seed=1337, val_dataset=val)
    tr = sum(1 for l in tl["labels"] if l in set(tc))
    va = sum(1 for l in vl["labels"] if l in set(tc))
    print(f"K={K}: in-class train={tr} val={va} classes={tc[:8]}...")

tc = choose_target_classes(train, num_classes=10, seed=1337, val_dataset=val)
s, _ = create_subset_sampler(train, max_num_classes=10, target_classes=tc, seed=1337)
print("sampler len:", len(s))
print("DONE")
