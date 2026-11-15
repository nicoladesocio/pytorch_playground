"""Exercise the resumable parquet download. Kanban `P3-11`.

Phase 1: an uninterrupted reference run, to get ground truth for "which rows does this
         seed select".
Phase 2: the identical call, SIGKILLed partway through the download, then re-run. The
         second call must print ``resuming after`` (proof it found and kept the
         checkpoint) and must land on *exactly* the same subset as phase 1.

Correctness criterion: the resumed dataset's label sequence and image paths must be
identical to the uninterrupted run's. A resume that silently re-draws the sample is
worse than no resume at all, because two "identical" training runs then differ.

This probe was written in HANDOFF_003 but **never executed**. It hardcoded
``/root/.hf_token`` (which does not exist in ``Dockerfile.gpu`` — the token comes from
the mounted HF cache), wrote ``_resume_test/`` and ``_check_big_child.py`` into the repo
root where they lingered as untracked files, and returned no exit code. All three fixed.

    python probes/check_resume.py
    python probes/check_resume.py --kill-after 45 --train-samples 8192
    python probes/check_resume.py --scratch /tmp/resume_scratch --keep

Exit code 0 only if the resume both happened and was exact.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

REPO_ROOT = _bootstrap.REPO_ROOT

# Written to the scratch dir, not the repo. It cannot import _bootstrap (that lives in
# probes/, which is not on the child's sys.path), so it inserts the repo root itself.
CHILD_SRC = '''
import json
import os
import sys

sys.path.insert(0, {repo!r})
from utils.datasets_utils import load_dataset_subset

cache_root, train_samples, val_samples, seed = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
train, val, _ = load_dataset_subset(train_samples=train_samples,
                                    val_samples=val_samples, seed=seed,
                                    cache_root=cache_root)
# The cache stores images undecoded, so the column is a {{'bytes','path'}} dict.
labels = list(train["label"])
paths = [r["image"]["path"] for r in train]
json.dump({{"labels": labels, "paths": paths}},
          open(os.path.join(cache_root, "fingerprint.json"), "w"))
print("CHILD DONE", len(train), len(val))
'''.format(repo=REPO_ROOT)


def run_child(child_path, cache_root, train_samples, val_samples, seed, kill_after=None):
    """Run one download in its own process group; optionally SIGKILL it mid-flight."""
    os.makedirs(cache_root, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, child_path, cache_root,
         str(train_samples), str(val_samples), str(seed)],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid,
    )
    lines, killed = [], False
    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        lines.append(line)
        if kill_after and (time.time() - t0) > kill_after:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print(f"  [killed after {time.time() - t0:.0f}s]")
            killed = True
            break
        if "CHILD DONE" in line:
            break
    proc.wait()
    return lines, killed


def interesting(lines):
    return [l for l in lines
            if "it [" not in l and "Generating" not in l
            and "examples/s]" not in l and "shards)" not in l and l.strip()]


def fingerprint(cache_root):
    path = os.path.join(cache_root, "fingerprint.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parts_on_disk(cache_root):
    """Where the resumable parquet parts and their checkpoint live.

    The manifest is ``_manifest.json`` (leading underscore, see
    ``datasets_utils._download_split_via_parquet``). Without it the part files are
    deliberately treated as untrusted and deleted, so its presence is what makes a
    resume possible at all.
    """
    found = []
    for dirpath, _dirs, files in os.walk(cache_root):
        parquets = sorted(f for f in files if f.endswith(".parquet"))
        if parquets:
            found.append((dirpath, parquets, "_manifest.json" in files))
    return found


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=None,
                    help="working dir; default = a temp dir outside the repo")
    ap.add_argument("--train-samples", type=int, default=8192)
    ap.add_argument("--val-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kill-after", type=float, default=45.0,
                    help="seconds before the phase-2 run is SIGKILLed")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch dir instead of deleting it")
    args = ap.parse_args()

    scratch = args.scratch or tempfile.mkdtemp(prefix="ppg_resume_",
                                               dir=tempfile.gettempdir())
    os.makedirs(scratch, exist_ok=True)
    child_path = os.path.join(scratch, "_check_big_child.py")
    with open(child_path, "w") as f:
        f.write(CHILD_SRC)
    print(f"scratch: {scratch}")
    print(f"request: {args.train_samples} train / {args.val_samples} val, seed={args.seed}")

    failures = 0

    # ------------------------------------------------------------- phase 1
    print("\n=== phase 1: uninterrupted reference run ===")
    ref_dir = os.path.join(scratch, "reference")
    shutil.rmtree(ref_dir, ignore_errors=True)
    t0 = time.time()
    lines, _ = run_child(child_path, ref_dir, args.train_samples,
                         args.val_samples, args.seed)
    print("\n".join(interesting(lines)))
    print(f"reference finished in {time.time() - t0:.0f}s")
    ref_fp = fingerprint(ref_dir)
    if ref_fp is None:
        print("FAIL: reference run produced no fingerprint; cannot judge the resume")
        return 1
    print(f"reference fingerprint: {len(ref_fp['labels'])} labels, "
          f"first5={ref_fp['labels'][:5]}")

    # ------------------------------------------------------------- phase 2
    print("\n=== phase 2: kill mid-download, then resume ===")
    res_dir = os.path.join(scratch, "resumed")
    shutil.rmtree(res_dir, ignore_errors=True)
    t0 = time.time()
    lines, killed = run_child(child_path, res_dir, args.train_samples,
                              args.val_samples, args.seed, kill_after=args.kill_after)
    print("\n".join(interesting(lines)))
    if not killed:
        print(f"FAIL: the run finished before {args.kill_after:.0f}s, so nothing was "
              "interrupted. Raise --train-samples or lower --kill-after.")
        return 1

    partial = parts_on_disk(res_dir)
    for dirpath, parquets, has_manifest in partial:
        print(f"partial parts in {os.path.relpath(dirpath, res_dir)}: "
              f"{len(parquets)} files, manifest={'yes' if has_manifest else 'NO'} "
              f"{parquets[:3]}{'...' if len(parquets) > 3 else ''}")
        if not has_manifest:
            print("    FAIL: parquet parts without a manifest cannot be resumed from")
            failures += 1
    if not partial:
        print("FAIL: the kill landed before any part file was written, so there is "
              "nothing to resume from. Raise --kill-after.")
        return 1

    print("\n--- resuming ---")
    t1 = time.time()
    lines, _ = run_child(child_path, res_dir, args.train_samples,
                         args.val_samples, args.seed)
    print("\n".join(interesting(lines)))
    print(f"resume finished in {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)")
    resumed = [l for l in lines if "resuming after" in l]
    print("RESUME EVIDENCE:", resumed[0].strip() if resumed
          else "!! no 'resuming after' line -- it re-downloaded from scratch")
    if not resumed:
        failures += 1

    res_fp = fingerprint(res_dir)
    if res_fp is None:
        print("FAIL: resumed run produced no fingerprint")
        return 1

    same_labels = ref_fp["labels"] == res_fp["labels"]
    same_paths = ref_fp["paths"] == res_fp["paths"]
    print(f"\nlabels identical to uninterrupted run: {same_labels}")
    print(f"paths  identical to uninterrupted run: {same_paths}")
    if not same_labels:
        i = first_difference(ref_fp["labels"], res_fp["labels"])
        print(f"    first label difference at index {i}: "
              f"reference={ref_fp['labels'][i]} resumed={res_fp['labels'][i]}")
    if not same_paths:
        i = first_difference(ref_fp["paths"], res_fp["paths"])
        print(f"    first path difference at index {i}:\n"
              f"      reference={ref_fp['paths'][i]}\n"
              f"      resumed  ={res_fp['paths'][i]}")
    failures += 0 if (same_labels and same_paths) else 1

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"(scratch kept at {scratch})")

    print()
    if failures:
        print("VERDICT: RESUME IS BROKEN -- the resumed subset differs from a clean run, "
              "or the checkpoint was not used.")
        return 1
    print("VERDICT: RESUME IS EXACT -- the interrupted download resumed from its "
          "checkpoint and reproduced the uninterrupted subset byte for byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
