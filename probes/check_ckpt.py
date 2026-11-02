"""P2-5 / P2-6 acceptance: checkpoint helpers are model-agnostic and resume works."""
import os
import tempfile

import torch

from utils.model_utils import (get_latest_checkpoint, save_model_checkpoint,
                               load_model_checkpoint)

ok = True

# 1. No model-specific defaults leak into the generic helpers.
import inspect
from utils import model_utils
for fn in ("get_latest_checkpoint", "save_model_checkpoint", "load_model_checkpoint"):
    sig = inspect.signature(getattr(model_utils, fn))
    for name, p in sig.parameters.items():
        if isinstance(p.default, str) and "alexnet" in p.default.lower():
            print(f"[P2-6] FAIL {fn}({name}) defaults to {p.default!r}")
            ok = False
src = open(model_utils.__file__, encoding="utf-8").read()
if "alexnet" in src.lower():
    print("[P2-6] FAIL 'alexnet' still appears in utils/model_utils.py")
    ok = False
else:
    print("[P2-6] OK   no AlexNet-specific strings in generic helpers")

# 2. weights_only=True is used for torch.load.
if "weights_only=True" in src:
    print("[P2-6] OK   torch.load uses weights_only=True")
else:
    print("[P2-6] FAIL torch.load without weights_only=True")
    ok = False

# 3. Filename derivation: two models into the same dir must not collide.
with tempfile.TemporaryDirectory() as d:
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    save_model_checkpoint(Fake(), d, filename="alexnet_final.ckpt")
    save_model_checkpoint(Fake(), d, filename="googlenet_final.ckpt")
    files = sorted(os.listdir(d))
    if len(files) == 2:
        print(f"[P2-6] OK   both artifacts coexist: {files}")
    else:
        print(f"[P2-6] FAIL collision, only {files}")
        ok = False

    m = load_model_checkpoint(Fake(), d, "googlenet_final.ckpt")
    print("[P2-6] OK   load_model_checkpoint round-trips:",
          tuple(m.state_dict()["lin.weight"].shape))

# 4. get_latest_checkpoint picks highest epoch then version (the real Lightning
#    naming, which relies on auto_insert_epoch inserting "epoch=").
with tempfile.TemporaryDirectory() as d:
    for n in ["alexnet_epoch_epoch=00.ckpt", "alexnet_epoch_epoch=00-v1.ckpt",
              "alexnet_epoch_epoch=03.ckpt", "alexnet_epoch_epoch=02-v10.ckpt",
              "googlenet_epoch_epoch=09.ckpt"]:
        open(os.path.join(d, n), "w").close()
    a = get_latest_checkpoint(d, model_prefix="alexnet_epoch")
    g = get_latest_checkpoint(d, model_prefix="googlenet_epoch")
    print(f"[P2-5] alexnet latest -> {os.path.basename(a) if a else None}")
    print(f"[P2-5] googlenet latest -> {os.path.basename(g) if g else None}")
    if a is None or os.path.basename(a) != "alexnet_epoch_epoch=03.ckpt":
        print("[P2-5] FAIL wrong epoch/version chosen")
        ok = False
    if g is None or os.path.basename(g) != "googlenet_epoch_epoch=09.ckpt":
        print("[P2-5] FAIL googlenet prefix lookup failed")
        ok = False
    if get_latest_checkpoint(d, model_prefix="nope_epoch") is not None:
        print("[P2-5] FAIL should return None for no matches")
        ok = False

# 5. --resume exists on both train scripts and fresh runs are unaffected.
import importlib
for mod_name, path in [("AlexNet.train", "AlexNet/train.py"),
                       ("GoogLeNet.train", "GoogLeNet/train.py")]:
    mod = importlib.import_module(mod_name)
    p = mod.build_arg_parser()
    ns = p.parse_args([])
    has = any("--resume" in a.option_strings for a in p._actions)
    print(f"[P2-5] {path}: --resume present={has}, default={getattr(ns, 'resume', 'MISSING')}")
    if not has or getattr(ns, "resume", True):
        ok = False

print("[P2-5/P2-6]", "PASS" if ok else "FAIL")
