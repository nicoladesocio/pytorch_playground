"""Acceptance check for P3-5: registry must not store host-specific absolute paths."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "trainer" / "registry.json"
BACKUP = ROOT / "_registry_backup.json"

if REG.exists():
    BACKUP.write_text(REG.read_text(encoding="utf-8"), encoding="utf-8")

try:
    for name, model_dir in [("alexnet", "AlexNet"), ("googlenet", "GoogLeNet")]:
        proc = subprocess.run(
            [sys.executable, "-m", "trainer", "register", "--model-name", name,
             "--model-dir", model_dir, "--force"],
            cwd=ROOT, capture_output=True, text=True,
        )
        print(f"[P3-5] register {name}: rc={proc.returncode}")
        if proc.returncode:
            print(proc.stdout[-500:], proc.stderr[-1500:])

    data = json.loads(REG.read_text(encoding="utf-8"))
    ok = True
    for name, rec in data.items():
        for key in ("model_dir", "module_file"):
            val = rec[key]
            if Path(val).is_absolute():
                print(f"[P3-5] FAIL {name}.{key} is absolute: {val}")
                ok = False
            else:
                print(f"[P3-5] OK   {name}.{key} relative: {val}")
            if not (ROOT / val).exists():
                print(f"[P3-5] FAIL {name}.{key} does not resolve: {ROOT / val}")
                ok = False

    for name in ("alexnet", "googlenet"):
        proc = subprocess.run(
            [sys.executable, "-m", "trainer", "train", name, "--train_samples", "64",
             "--val_samples", "16", "--dry-run"],
            cwd=ROOT, capture_output=True, text=True,
        )
        line = [l for l in proc.stdout.splitlines() if "train.py" in l]
        print(f"[P3-5] dry-run {name}: rc={proc.returncode} {line[0][:110] if line else proc.stderr[-400:]}")
        if proc.returncode != 0:
            ok = False
    print("[P3-5]", "PASS" if ok else "FAIL")
finally:
    if BACKUP.exists():
        REG.write_text(BACKUP.read_text(encoding="utf-8"), encoding="utf-8")
        BACKUP.unlink()
