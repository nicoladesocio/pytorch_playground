"""Is the augmentation destroying the signal at small --image_size?

ImageNetSubsetDataset hardcodes transforms.Resize(256) then RandomCrop(image_size).
At image_size=224 the crop keeps (224/256)^2 = 77% of the area, which is the paper's
recipe. At image_size=96 it keeps (96/256)^2 = 14% -- a random postage stamp, which
for a 10-class / ~90-image probe can easily contain no class-identifying content.

Test: overfit the SAME real images with (a) the training transform and (b) the eval
transform. If (b) memorises and (a) does not, the crop is the blocker, not the model.
"""
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from datasets import load_from_disk

from AlexNet.model import AlexNet
from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset
from utils.datasets_utils import create_subset_sampler

SRC = "/workspace/Data/datasets/ILSVRC__imagenet-1k_4096t_512v_auto_a09e9cc1ed88"
K = 10
IMG = 96
B = 16
STEPS = 80
LR = 1e-3

train = load_from_disk(SRC + "/train")
val = load_from_disk(SRC + "/validation")
sampler, target_classes = create_subset_sampler(
    train, max_num_classes=K, seed=42, val_dataset=val, num_samples=None)
target_classes = sorted(target_classes)
in_class = len(list(sampler))
print(f"target_classes={target_classes}\nin-class images={in_class}")

chance = float(torch.log(torch.tensor(float(K))))


def run(tag, is_training):
    ds = ImageNetSubsetDataset(train, image_size=IMG, is_training=is_training,
                               target_classes=target_classes)
    loader = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=2,
                        drop_last=False, persistent_workers=True)
    net = AlexNet(num_classes=K, image_size=IMG)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rows = []
    t0 = time.time()
    step = 0
    while step < STEPS:
        for images, labels in loader:
            step += 1
            opt.zero_grad(set_to_none=True)
            out = net(images)
            loss = F.cross_entropy(out, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step in (1, 10, 20, 40, 60, STEPS):
                acc = (out.argmax(1) == labels.long()).float().mean().item()
                rows.append((step, loss.item(), acc))
            if step >= STEPS:
                break
    print(f"\n{tag}: chance={chance:.4f}  {STEPS} steps in {time.time()-t0:.0f}s")
    for s, l, a in rows:
        print(f"  step {s:3d} loss {l:7.4f} acc {a:.3f}")
    first, last = rows[0][1], rows[-1][1]
    print(f"  -> {first:.4f} -> {last:.4f} : "
          f"{'MEMORISED' if last < chance * 0.5 else 'not memorised'}")
    return first, last


run("TRAIN transform (Resize256+RandomCrop96)", True)
run("EVAL  transform (Resize256+CenterCrop96)", False)
