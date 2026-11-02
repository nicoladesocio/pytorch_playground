"""Build the canonical subset used by train_final.py and report its statistics.

Run from anywhere:  python probes/check_big.py

``SEED`` must match the ``"seed"`` baked into ``AlexNet/train_final.py`` and
``GoogLeNet/train_final.py``: it is part of the cache key, so a probe using any
other seed builds a *different* 5.7 GB subset and tells you nothing about the
cache the deliverable will actually load.
"""
import time

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from utils.datasets_utils import (load_dataset_subset, label_distribution,
                                  choose_target_classes, create_subset_sampler)

T, V, SEED = 51200, 2048, 42
t0 = time.time()
train, val, _ = load_dataset_subset(train_samples=T, val_samples=V, seed=SEED)
print(f"elapsed={time.time()-t0:.1f}s train={len(train)} val={len(val)}")

tl = label_distribution(train)
vl = label_distribution(val)
print("train classes:", len(tl))
print("val classes:", len(vl))
q = sorted(tl.values())
print("per-class train min/p25/med/p75/max:",
      q[0], q[len(q)//4], q[len(q)//2], q[3*len(q)//4], q[-1])
qv = sorted(vl.values())
print("per-class val   min/p25/med/p75/max:",
      qv[0], qv[len(qv)//4], qv[len(qv)//2], qv[3*len(qv)//4], qv[-1])

for K in (10, 64, 128):
    tc = choose_target_classes(train, num_classes=K, seed=SEED, val_dataset=val)
    ts = set(tc)
    tr = sum(c for l, c in tl.items() if l in ts)
    va = sum(c for l, c in vl.items() if l in ts)
    print(f"K={K}: in-class train={tr} val={va} classes={sorted(tc)[:8]}...")

tc = choose_target_classes(train, num_classes=10, seed=SEED, val_dataset=val)
s, _ = create_subset_sampler(train, max_num_classes=10, target_classes=tc, seed=SEED)
print("sampler len:", len(s))
print("DONE")
