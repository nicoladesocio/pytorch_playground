"""Does Lightning aggregate a logged Metric correctly, or mean-average the batches?

  A) self.log("x", metric(preds, target))  -> passes the returned tensor
  B) self.log("x", metric)                 -> passes the Metric object

Lightning reduces a plain tensor with reduce_fx="mean" but compute()s a Metric object
at epoch end. With unequal batch sizes those differ, so this has to be measured.

Batches of size 8 / 8 / 4 (unequal, which is exactly what a WeightedRandomSampler
with drop_last=False produces). Correctness pattern: all, none, all.
    true global accuracy  = 12/20        = 60.0 %
    mean of per-batch pct = (100+0+100)/3 = 66.667 %
"""

import sys

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

import lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import PercentageAccuracy

N = 20
# predicted class per sample, and the true class
pred = torch.tensor([0] * 8 + [1] * 8 + [0] * 4)   # batches: all-correct, all-wrong, all-correct
target = torch.tensor([0] * 20)
assert ((pred == target)[:8]).all() and not ((pred == target)[8:16]).any()
assert ((pred == target)[16:]).all()


class Net(pl.LightningModule):
    def __init__(self, style):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.crit = nn.CrossEntropyLoss()
        self.style = style
        self.acc = PercentageAccuracy(num_classes=2)

    def training_step(self, batch, idx):
        p, y = batch
        logits = torch.nn.functional.one_hot(p, 2).float() * 10.0 + self.dummy
        if self.style == "A":
            self.log("acc", self.acc(logits, y), on_step=False, on_epoch=True)
        else:
            self.acc.update(logits, y)
            self.log("acc", self.acc, on_step=False, on_epoch=True)
        return self.crit(logits, y) + 0 * self.dummy

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


ds = TensorDataset(pred, target)


class FixedBatches(torch.utils.data.Sampler):
    """Yield pre-built batches so the sizes are genuinely unequal."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


for style in ("A", "B"):
    net = Net(style)
    dl = DataLoader(ds, batch_sampler=FixedBatches([[0] * 8, [8] * 8, [16] * 4]))
    tr = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                    max_epochs=3, enable_progress_bar=False,
                    enable_model_summary=False, num_sanity_val_steps=0)
    tr.fit(net, dl)
    sd = net.acc.state_dict()
    cm = sd.get("confmat")
    total = int(cm.sum()) if cm is not None else -1
    print(f"style {style}: acc = {net.trainer.callback_metrics['acc'].item():.4f} "
          f"confmat_sum_after_fit={total} (20 => reset per epoch, 60 => not reset)")

print("true global = 60.0000 ; mean-of-batches = 66.6667")
