"""Exercise the resumable parquet download.

Phase 1: start a download in a subprocess and kill it partway through.
Phase 2: re-run the identical call and confirm it resumes (prints 'resuming after')
         and lands on exactly the same subset as an uninterrupted run.

Correctness criterion: the resumed dataset must be byte-identical (same label
sequence) to a dataset produced by one uninterrupted run with the same seed.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

ROOT = "/workspace/_resume_test"
CHILD = "/workspace/_check_big_child.py"

CHILD_SRC = '''
import sys
import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from datasets import load_from_disk
from utils.datasets_utils import load_dataset_subset
import os
cache_root = sys.argv[1]
train, val, _ = load_dataset_subset(train_samples=8192, val_samples=512, seed=7,
                                    cache_root=cache_root)
labels = list(train["label"])
paths = [r["image"]["path"] for r in train]
import json
json.dump({"labels": labels, "paths": paths},
          open(os.path.join(cache_root, "fingerprint.json"), "w"))
print("CHILD DONE", len(train), len(val))
'''

with open(CHILD, "w") as f:
    f.write(CHILD_SRC)


def fresh(tag):
    d = os.path.join(ROOT, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


def run_child(cache_root, env_extra=None, kill_after=None):
    env = dict(os.environ)
    env["HF_TOKEN"] = open("/root/.hf_token").read().strip()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen([sys.executable, CHILD, cache_root], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=os.setsid)
    lines = []
    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
        if kill_after and (time.time() - t0) > kill_after:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print(f"  [killed after {time.time()-t0:.0f}s]")
            break
        if "CHILD DONE" in line:
            break
    proc.wait()
    return lines


print("=== phase 1: uninterrupted reference run ===")
ref_dir = fresh("reference")
t0 = time.time()
lines = run_child(ref_dir)
print("\n".join(l for l in lines if "it [" not in l and "Generating" not in l))
print(f"reference finished in {time.time()-t0:.0f}s")
ref_fp = json.load(open(os.path.join(ref_dir, "fingerprint.json")))
print(f"reference fingerprint: {len(ref_fp['labels'])} labels, "
      f"first5={ref_fp['labels'][:5]}")

print("\n=== phase 2: kill mid-download, then resume ===")
res_dir = fresh("resumed")
t0 = time.time()
lines = run_child(res_dir, kill_after=90)
print("\n".join(l for l in lines if "it [" not in l and "Generating" not in l))
parts_dir = None
for dirpath, dirs, files in os.walk(res_dir):
    if any(f.endswith(".parquet") for f in files):
        parts_dir = dirpath
        break
print(f"partial parts dir: {parts_dir}")
if parts_dir:
    print("  files:", sorted(os.listdir(parts_dir)))

print("\n--- resuming ---")
t1 = time.time()
lines = run_child(res_dir)
resumed = [l for l in lines if "resuming after" in l]
print("\n".join(l for l in lines if "it [" not in l and "Generating" not in l))
print(f"resume finished in {time.time()-t1:.0f}s (total {time.time()-t0:.0f}s)")
print("RESUME EVIDENCE:", resumed or "!! no 'resuming after' line -- did NOT resume")

res_fp = json.load(open(os.path.join(res_dir, "fingerprint.json")))
same_labels = ref_fp["labels"] == res_fp["labels"]
same_paths = ref_fp["paths"] == res_fp["paths"]
print(f"\nlabels identical to uninterrupted run: {same_labels}")
print(f"paths  identical to uninterrupted run: {same_paths}")
print("VERDICT:", "RESUME IS EXACT" if (same_labels and same_paths)
      else "RESUME DIVERGED -- resumed subset differs from a clean run")
