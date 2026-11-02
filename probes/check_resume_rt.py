"""P2-5: real resume round-trip -- does --resume restore epoch/optimizer/scheduler?"""
import glob
import os
import shutil
import sys

import torch
import lightning as pl

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from AlexNet.model import AlexNetLightning  # noqa: E402


class Tiny(torch.utils.data.Dataset):
    def __init__(self, n=64, nc=4):
        self.x = torch.randn(n, 3, 96, 96)
        self.y = torch.randint(0, nc, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


D = "_resume_roundtrip"
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D, exist_ok=True)
ckpt = os.path.join(D, "ck")

dl = torch.utils.data.DataLoader(Tiny(), batch_size=16)


TARGET_CLASSES = [3, 17, 42, 91]


def build():
    # train.py always passes a sorted target_classes list; mirror that so the
    # on_load_checkpoint compatibility guard sees a realistic checkpoint.
    return AlexNetLightning(num_classes=4, target_classes=list(TARGET_CLASSES),
                            image_size=96, lr=0.01)


t1 = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False,
                enable_checkpointing=False, enable_progress_bar=False,
                default_root_dir=D)
m1 = build()
t1.fit(m1, dl, dl)
ep_after_first = t1.current_epoch
print("phase1 current_epoch =", ep_after_first)

# Persist via ModelCheckpoint the same way train.py does.
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
t2 = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_progress_bar=False,
                default_root_dir=D,
                callbacks=[ModelCheckpoint(dirpath=ckpt, every_n_epochs=1,
                                           filename="alexnet_epoch_{epoch:02d}",
                                           save_top_k=-1)])
m2 = build()
t2.fit(m2, dl, dl)
files = sorted(glob.glob(os.path.join(ckpt, "*.ckpt")))
print("checkpoints:", [os.path.basename(f) for f in files])

from utils.model_utils import get_latest_checkpoint  # noqa: E402
latest = get_latest_checkpoint(ckpt, model_prefix="alexnet_epoch")
print("latest ->", os.path.basename(latest) if latest else None)
if latest is None:
    print("[P2-5] FAIL resume cannot find a checkpoint written by train.py's callback")
    raise SystemExit(1)

t3 = pl.Trainer(max_epochs=4, accelerator="cpu", logger=False,
                enable_checkpointing=False, enable_progress_bar=False,
                default_root_dir=D)
m3 = build()
t3.fit(m3, dl, dl, ckpt_path=latest)
print("phase3 current_epoch =", t3.current_epoch)
ok = t3.current_epoch >= 2
# Optimizer/scheduler state must have survived.
last = m3.trainer.optimizers[0]
has_state = bool(last.state) and all("momentum_buffer" in v for v in last.state.values()
                                     if isinstance(v, dict))
print("optimizer state restored =", has_state)
sched = m3.trainer.lr_scheduler_configs
print("scheduler restored =", bool(sched),
      "last_epoch =", sched[0].scheduler.last_epoch if sched else None)
ok = ok and has_state and bool(sched)

# 6. The compatibility guard must still reject a genuinely OLD checkpoint (the
#    'target_classes' key absent entirely), while accepting a current-code
#    checkpoint where the key is present.
import lightning.pytorch as _pl  # noqa: E402
old = torch.load(latest, map_location="cpu", weights_only=False)
old["hyper_parameters"].pop("target_classes", None)
old_path = os.path.join(D, "old.ckpt")
torch.save(old, old_path)
raised = False
try:
    t4 = _pl.Trainer(max_epochs=3, accelerator="cpu", logger=False,
                     enable_checkpointing=False, enable_progress_bar=False,
                     default_root_dir=D)
    t4.fit(build(), dl, dl, ckpt_path=old_path)
except ValueError as e:
    raised = "target_classes" in str(e)
print("legacy checkpoint rejected =", raised)
if not raised:
    print("[P2-5] FAIL legacy (pre-remapping) checkpoint was accepted")
    ok = False

print("[P2-5] resume round-trip", "PASS" if ok else "FAIL")
