"""Verify that the label tensor a DataLoader yields matches the actual image.

If the target_classes remap is inconsistent between the image and the label, the
model can never beat chance no matter how good the architecture -- which looks
exactly like 'loss decreases very slowly, accuracy stuck at chance'.

ImageNet-1k filenames encode the synset (e.g. n02978881_...), and the dataset's
`label` column is the integer class index. We check the mapping is self-consistent
across the whole cached subset, then check what the DataLoader actually yields.
"""
import collections
import sys

from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from utils.datasets_utils import (
    choose_target_classes,
    create_subset_sampler,
)
from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset
from datasets import load_from_disk

SRC = "/workspace/Data/datasets/ILSVRC__imagenet-1k_4096t_512v_auto_a09e9cc1ed88"

train = load_from_disk(SRC + "/train")
val = load_from_disk(SRC + "/validation")
print("train features:", dict(train.features))
print("first row:", {k: (type(v).__name__ if not isinstance(v, (int, str)) else v)
                     for k, v in train[0].items()})

# 1. Is `label` consistent with the synset prefix in the filename?
pairs = collections.Counter()
bad = 0
for i in range(min(600, len(train))):
    row = train[i]
    path = row["image"]["path"] if isinstance(row["image"], dict) else row["image"]
    synset = path.split("_")[0]
    pairs[(row["label"], synset)] += 1
    if not synset.startswith("n0") and not synset.startswith("n1"):
        bad += 1
print(f"\nchecked 600 rows: {len(pairs)} distinct (label, synset) pairs, bad names={bad}")
multi = {lbl: {s for (l, s) in pairs if l == lbl}
         for lbl in {l for (l, _s) in pairs}}
ambiguous = {l: s for l, s in multi.items() if len(s) > 1}
print(f"labels mapping to >1 synset: {len(ambiguous)}")
for lbl, synsets in list(ambiguous.items())[:5]:
    print("   ", lbl, sorted(synsets)[:4])

# 2. What does the actual training pipeline yield?
K = 10
targets = choose_target_classes(train, num_classes=K, seed=42, val_dataset=val)
print(f"\nchosen target_classes: {targets}")
ds, _idx2cls = create_subset_sampler(train, target_classes=targets, num_samples=200)
print("dataset type:", type(ds).__name__)
loader = DataLoader(ds, batch_size=16, num_workers=0)
images, labels = next(iter(loader))
print("batch images:", tuple(images.shape), images.dtype, "labels:", labels.tolist())

# 3. Cross-check: for each yielded label, what synset does the filename say?
#    create_subset_sampler keeps the original row, so re-read the same rows.
print("\nlabel -> synsets actually yielded:")
seen = collections.defaultdict(set)
for i in range(len(ds)):
    item = ds[i]
    img = item[0]
    lbl = int(item[1])
    path = img["path"] if isinstance(img, dict) else img
    seen[lbl].add(str(path).split("_")[0])
for lbl in sorted(seen):
    print(f"  label {lbl}: {sorted(seen[lbl])}")
print("\nIf a label maps to MANY synsets, the remap is wrong (or classes are mixed).")
print("target_classes list is the ground truth for label -> synset.")
