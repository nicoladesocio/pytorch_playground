"""Final gate: re-verify every card claimed DONE in HANDOFF_003."""
import importlib
import inspect
import subprocess
import sys
import os

ok = True
def chk(label, cond, extra=""):
    global ok
    print(f"[{'OK  ' if cond else 'FAIL'}] {label} {extra}")
    if not cond:
        ok = False

# P1-1 import safety
for m in ("AlexNet.train", "GoogLeNet.train"):
    src = open(f"{m.replace('.', '/')}.py", encoding="utf-8").read()
    body = src.split('if __name__')[0]
    chk(f"P1-1 {m}: no Trainer( at module level", "pl.Trainer(" not in body)

# P1-2 imports
import GoogLeNet.model as gm, GoogLeNet.train as gt, AlexNet.model as am
chk("P1-2 GoogLeNet.model/train importable from repo root", True)
for f in ("AlexNet/model.py", "GoogLeNet/model.py", "AlexNet/train.py", "GoogLeNet/train.py"):
    src = open(f, encoding="utf-8").read()
    chk(f"P1-2 {f}: uses 'import lightning as pl'", "import lightning as pl" in src)
    if f.endswith("model.py"):
        chk(f"P1-2 {f}: no bare 'import model'/'from inception'",
            "\nimport model" not in src and "\nfrom inception" not in src)

# P1-4 portable paths
import utils.runtime as rt
chk("P1-4 repo_root/default_path exist", hasattr(rt, "repo_root") and hasattr(rt, "default_path"))
for f in ("AlexNet/train.py", "GoogLeNet/train.py"):
    src = open(f, encoding="utf-8").read()
    chk(f"P1-4 {f}: no /workspace default", '"/workspace' not in src and "'/workspace" not in src)

# P1-6
chk("P1-6 Data/datasets/.gitkeep exists", os.path.exists("Data/datasets/.gitkeep"))

# P1-7
import tomllib
with open("pyproject.toml", "rb") as fh:
    pk = tomllib.load(fh)["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
chk("P1-7 trainer in wheel packages", any("trainer" in str(p) and "trainer_v2" not in str(p) for p in pk), pk)

# P2-1 no prints in steps
for f in ("AlexNet/model.py", "GoogLeNet/model.py"):
    src = open(f, encoding="utf-8").read()
    for step in ("training_step", "validation_step"):
        i = src.find(f"def {step}")
        seg = src[i:i + 2000]
        seg = seg[:seg.find("\n    def ", 10)]
        chk(f"P2-1 {f} {step}: no print(", "print(" not in seg)

# P2-2 seeding
for f in ("AlexNet/train.py", "GoogLeNet/train.py"):
    src = open(f, encoding="utf-8").read()
    chk(f"P2-2 {f}: seed_everything + --seed", "seed_everything" in src and "--seed" in src)

# P2-3 real CLI args
for m in ("AlexNet.train", "GoogLeNet.train"):
    p = importlib.import_module(m).build_arg_parser()
    opts = {o for a in p._actions for o in a.option_strings}
    need = {"--max_epochs", "--lr", "--optimizer", "--scheduler"}
    chk(f"P2-3 {m}: {sorted(need)}", need <= opts, sorted(need - opts))

# P2-4 single source of truth
src = open("AlexNet/train.py", encoding="utf-8").read()
chk("P2-4 AlexNet sampler uses args.num_classes", "max_num_classes=args.num_classes" in src)

# P2-5 resume
for m in ("AlexNet.train", "GoogLeNet.train"):
    p = importlib.import_module(m).build_arg_parser()
    opts = {o for a in p._actions for o in a.option_strings}
    chk(f"P2-5 {m}: --resume", "--resume" in opts)
chk("P2-5 AlexNet guard tests key presence not None",
    '"target_classes" not in hparams' in open("AlexNet/model.py", encoding="utf-8").read())

# P2-6
from utils import model_utils
src = open(model_utils.__file__, encoding="utf-8").read()
chk("P2-6 no alexnet strings", "alexnet" not in src.lower())
chk("P2-6 weights_only=True", "weights_only=True" in src)
for fn in ("get_latest_checkpoint", "save_model_checkpoint", "load_model_checkpoint"):
    sig = inspect.signature(getattr(model_utils, fn))
    chk(f"P2-6 {fn} has required filename/prefix",
        all(p.default is inspect.Parameter.empty for p in sig.parameters.values()))

# P2-7
isrc = open("GoogLeNet/inception.py", encoding="utf-8").read()
i = isrc.find("isinstance(m, torch.nn.Linear)")
lin_block = isrc[i:i + 200] if i >= 0 else ""
chk("P2-7 Linear init not kaiming fan_out",
    "xavier" in lin_block or "kaiming_normal_(m.weight, mode='fan_in'" in lin_block,
    repr(lin_block[:80]))
gsrc = open("GoogLeNet/model.py", encoding="utf-8").read()
chk("P2-7 aux weight configurable", "aux_loss_weight" in gsrc)
chk("P2-7 aux losses logged separately", 'train_loss_aux1' in gsrc and 'train_loss_aux2' in gsrc)

# P2-10
for m in ("AlexNet.train", "GoogLeNet.train"):
    p = importlib.import_module(m).build_arg_parser()
    opts = {o for a in p._actions for o in a.option_strings}
    chk(f"P2-10 {m}: --train_samples_per_epoch", "--train_samples_per_epoch" in opts)
asrc = open("AlexNet/train.py", encoding="utf-8").read()
chk("P2-10 val loader drop_last=False", "drop_last=False" in asrc)

# P2-13
chk("P2-13 flatten derived from conv stack", "_flatten_features" in open("AlexNet/model.py", encoding="utf-8").read())
_am = [l for l in open("AlexNet/model.py", encoding="utf-8").read().splitlines()
       if "256 * 6 * 6" in l and not l.strip().startswith("#") and "Hardcoding" not in l]
chk("P2-13 no hardcoded 256*6*6 in code", not _am, _am)

# P3-5
from trainer.trainer import _rel_to_root, _abs_from_root, PROJECT_ROOT
chk("P3-5 rel/abs helpers round-trip",
    _abs_from_root(_rel_to_root(PROJECT_ROOT / "AlexNet")) == (PROJECT_ROOT / "AlexNet").resolve())

print("\nFINAL:", "ALL PASS" if ok else "FAILURES PRESENT")
