"""Can each network actually *learn from real images*, on CPU and on GPU?

This is the question the repo has been chasing since HANDOFF_001, and it is the one the
definition of done in `goal.md` is written against: loss must fall **below chance** and
validation accuracy must rise **clearly above chance**.

Two earlier probes answer it only partially:

* `check_overfit2.py` uses synthetic Gaussian noise, so it proves the architecture and
  initialisation are sound but says nothing about the data path.
* `--smoke` runs the real pipeline but only for 2 epochs on 1024 samples, which is too
  short to leave chance level; HANDOFF_003 §6.8's smoke numbers sit exactly on chance and
  were briefly misread as failure.

This probe closes the gap: real cached images, the real `ImageNetSubsetDataset` and the
real samplers, and enough steps to actually move. It sweeps the full matrix
{AlexNet, GoogLeNet} x {cpu, cuda} x {the caches present on disk} and prints one table.

    python probes/check_learn.py                       # every cache, best device each
    python probes/check_learn.py --device cuda
    python probes/check_learn.py --cache <dir> --steps 400

A cell PASSES when train loss ends below 0.6x that network's own chance level (see
`check_overfit2.chance_loss` and kanban `P2-17` -- GoogLeNet's combined loss starts at
1.6*ln(K), so comparing it to ln(K) calls a healthy network broken).

Exit code 0 only if every requested cell passed.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

REPO_ROOT = _bootstrap.REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, "probes"))

from check_overfit2 import chance_loss  # noqa: E402  (shared chance rule, P2-17)

AUX_WEIGHT = 0.3


def find_caches(cache_root=None, only=None):
    """Cached subsets under the repo cache root, newest first."""
    root = cache_root or os.path.join(REPO_ROOT, "Data", "datasets")
    cands = sorted(glob.glob(os.path.join(root, "ILSVRC__imagenet-1k_*")),
                   key=os.path.getmtime, reverse=True)
    cands = [c for c in cands
             if os.path.isdir(os.path.join(c, "train"))
             and os.path.isdir(os.path.join(c, "validation"))]
    if only:
        cands = [c for c in cands if only in os.path.basename(c)]
    return cands


def load_model_module(dir_name, module_name):
    """Import AlexNet/model.py or GoogLeNet/model.py under a unique module name.

    Both files do a bare ``import model`` / ``from inception import ...``, so importing
    them both normally makes the second silently reuse the first (see AGENT_INSTRUCTIONS
    section 3.5). Give each its own module name and put its own folder on sys.path.
    """
    import importlib.util

    model_dir = os.path.join(REPO_ROOT, dir_name)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(model_dir, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_batch(cache, classes, batch, image_size, seed):
    """One fixed real batch: the same images and labels for every cell in the matrix.

    Returns ``(pils, y, K, target_classes)``. The chosen rows are decoded to PIL RGB
    **once** and the caller applies whichever transform it needs, so the eval and train
    arms of the `P2-15` A/B are guaranteed to see the same 32 images -- the only
    difference between them is the transform. Re-decoding through the HF dataset on
    every step would make the train arm measure JPEG decode speed rather than
    augmentation.

    ``image_size`` is not used here any more; it belongs to the transform, which the
    caller owns. It is kept in the signature because callers pass it positionally.
    """
    from datasets import load_from_disk

    from utils.datasets_utils import choose_target_classes, create_subset_sampler

    train = load_from_disk(os.path.join(cache, "train"))
    val = load_from_disk(os.path.join(cache, "validation"))
    targets = choose_target_classes(train, num_classes=classes, seed=seed,
                                    val_dataset=val)
    sampler, target_classes = create_subset_sampler(
        train, max_num_classes=classes, target_classes=targets, seed=seed)

    # The sampler draws from the *global* torch RNG, so the caller must have seeded it
    # (see main()). list(sampler) is called exactly once per cache -- calling it per arm
    # would draw a different 32 images for each arm and silently turn the A/B into a
    # comparison of two different batches.
    indices = [int(i) for i in list(sampler)[:batch]]

    pils = []
    for idx in indices:
        raw = train[idx]["image"]
        if isinstance(raw, dict) or isinstance(raw, (bytes, bytearray)):
            payload = raw["bytes"] if isinstance(raw, dict) else raw
            import io
            from PIL import Image as PILImage
            pils.append(PILImage.open(io.BytesIO(payload)))
        else:
            pils.append(raw)
    pils = [p.convert("RGB") for p in pils]

    # Labels come from the same remapping the training path uses, so the probe and
    # train.py agree on what index a class has.
    class_to_idx = {c: i for i, c in enumerate(target_classes)}
    y = torch.tensor([class_to_idx[train[i]["label"]] for i in indices], dtype=torch.long)

    del train, val
    return pils, y, len(target_classes), target_classes


def make_arm_transform(cache, target_classes, image_size, arm):
    """The transform for one arm, taken from the real dataset class.

    Using ``ImageNetSubsetDataset``'s own ``self.transform`` (rather than rebuilding a
    Compose here) is the point: the probe then measures the transform the training run
    actually applies, and cannot drift from it.
    """
    from datasets import load_from_disk

    from custum_datasets.imagenet_subset_dataset import ImageNetSubsetDataset

    train = load_from_disk(os.path.join(cache, "train"))
    ds = ImageNetSubsetDataset(train, image_size=image_size,
                               is_training=(arm == "train"),
                               target_classes=target_classes)
    transform = ds.transform
    del train
    return transform


def run_cell(name, make, x, y, K, device, steps, lr, pils=None, transform=None,
             seed=42):
    """Overfit one fixed batch.

    When ``pils``/``transform`` are given the batch is re-augmented from the original
    PIL images at every step (the training arm); otherwise the fixed tensor ``x`` is
    reused verbatim (the eval arm).
    """
    # Seed *before* the network is constructed. It used to be seeded after `make()`,
    # so the initial weights came from whatever RNG state the previous cell left
    # behind: the same command gave PASS and FAIL on different runs of the same cell,
    # and the sweep was not reproducible.
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed_all(seed)
    net = make().to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    chance = chance_loss(name, K)
    augmented = pils is not None
    if not augmented:
        xc, yc = x.to(device), y.to(device)

    first = last = None
    t0 = time.time()
    for step in range(1, steps + 1):
        if augmented:
            xc = torch.stack([transform(p) for p in pils]).to(device)
            yc = y.to(device)
        opt.zero_grad(set_to_none=True)
        if name == "GoogLeNet":
            out, aux2, aux1 = net(xc)
            loss = (F.cross_entropy(out, yc)
                    + AUX_WEIGHT * F.cross_entropy(aux1, yc)
                    + AUX_WEIGHT * F.cross_entropy(aux2, yc))
        else:
            out = net(xc)
            loss = F.cross_entropy(out, yc)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
        if not math.isfinite(last):
            return {"status": "NaN", "chance": chance, "first": first, "last": last,
                    "seconds": time.time() - t0}

    # Report the per-sample loss distribution, not just the mean. The mean is what the
    # pass rule uses (it is the documented rule, and it is what `goal.md`'s definition
    # of done is written against), but on a 32-image batch a single sample the network
    # refuses to fit contributes its whole loss to that mean: a cell that has memorised
    # 31 of 32 images reads as loss 3.41 / acc 0.97, which looks like a failure and is
    # not one. median + max make that visible instead of mysterious.
    per = F.cross_entropy(out, yc, reduction="none").detach()
    acc = (out.argmax(1) == yc).float().mean().item()
    return {
        "status": "PASS" if last < chance * 0.6 else "FAIL",
        "chance": chance, "first": first, "last": last, "grad": grad, "acc": acc,
        "median": per.median().item(), "max": per.max().item(),
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "both"])
    ap.add_argument("--cache", default=None, help="substring filter on the cache dir name")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=160,
                    help="keep >=128 or GoogLeNet's aux heads collapse (P2-16)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", default="eval", choices=["eval", "train", "both"],
                    help="eval = fixed pixels (CenterCrop); train = re-augment the same "
                         "images every step (Resize256+RandomCrop+flip+ColorJitter). "
                         "'both' runs the P2-15 A/B on identical images.")
    ap.add_argument("--is_training", action="store_true",
                    help="shorthand for --arm train")
    args = ap.parse_args()
    if args.is_training:
        args.arm = "train"

    # Seed before anything consumes the RNG. build_batch() calls list(sampler) on a
    # WeightedRandomSampler, which draws from the *global* torch RNG -- so without this
    # the 32 images the whole sweep memorises were re-drawn on every invocation and
    # `--seed` did not actually pin the batch. A cell could then pass and fail between
    # runs for a reason that has nothing to do with the network.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.image_size < 128:
        print(f"WARNING: image_size={args.image_size} collapses GoogLeNet's aux heads "
              "(P2-16); its verdict is not a fair test of the architecture.")

    if args.device == "auto":
        devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    elif args.device == "both":
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            sys.exit("--device cuda requested but torch.cuda.is_available() is False. "
                     "Run probes/check_hardware.py to find out why.")
        devices = [args.device]

    caches = find_caches(args.cache_root, args.cache)
    if not caches:
        sys.exit("No cached subset found. Build one:  python probes/check_big.py")

    alex_mod = load_model_module("AlexNet", "ppg_alexnet_model")
    goog_mod = load_model_module("GoogLeNet", "ppg_googlenet_model")
    makers = {
        "AlexNet": lambda K, IMG: alex_mod.AlexNet(num_classes=K, image_size=IMG),
        "GoogLeNet": lambda K, IMG: goog_mod.GoogLeNet(num_classes=K, image_size=IMG),
    }

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"torch {torch.__version__}   cuda available: {torch.cuda.is_available()}"
          + (f"   gpu: {gpu_name}" if gpu_name else ""))
    print(f"devices: {devices}")
    print(f"caches:  {[os.path.basename(c) for c in caches]}")

    rows = []
    arms = ["eval", "train"] if args.arm == "both" else [args.arm]
    for cache in caches:
        tag = os.path.basename(cache)
        # The batch is built ONCE per cache, outside the arm loop. Building it per arm
        # re-drew the sampler and gave each arm a different 32 images, which turns the
        # P2-15 A/B into a comparison of two different batches -- exactly the mistake
        # this probe was written to avoid.
        pils, y, K, target_classes = build_batch(
            cache, args.classes, args.batch, args.image_size, args.seed)
        for arm in arms:
            transform = make_arm_transform(cache, target_classes, args.image_size, arm)
            x = torch.stack([transform(p) for p in pils])
            print(f"\n=== {tag}  arm={arm}  ({x.shape[0]} images, {K} classes, "
                  f"{tuple(x.shape[2:])}, seed={args.seed}) ===")
            for name in ("AlexNet", "GoogLeNet"):
                for device in devices:
                    make = lambda K=K, name=name: makers[name](K, args.image_size)
                    r = run_cell(name, make, x, y, K, device, args.steps, args.lr,
                                 pils=(pils if arm == "train" else None),
                                 transform=transform, seed=args.seed)
                    r.update(model=name, device=device, cache=tag, arm=arm)
                    rows.append(r)
                    print(f"  {name:<10} {device:<5} {arm:<6} "
                          f"chance {r['chance']:.4f}   "
                          f"{r['first']:.4f} -> {r['last']:.4f}   "
                          f"med {r.get('median', float('nan')):.4f} "
                          f"max {r.get('max', float('nan')):.3f}   "
                          f"acc {r.get('acc', float('nan')):.3f}   "
                          f"{r['seconds']:5.1f}s   {r['status']}")
            del x, transform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del pils, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print(f"{'model':<11}{'device':<8}{'arm':<7}{'cache':<40}{'result'}")
    print("-" * 92)
    for r in rows:
        print(f"{r['model']:<11}{r['device']:<8}{r['arm']:<7}"
              f"{r['cache'][:39]:<40}{r['status']}")
    print("=" * 92)

    failed = [r for r in rows if r["status"] != "PASS"]
    if failed:
        print(f"\nOVERALL: {len(failed)}/{len(rows)} cell(s) did not beat chance: "
              + ", ".join(f"{r['model']}/{r['device']}/{r['arm']}/{r['cache']}"
                          for r in failed))
        return 1
    print(f"\nOVERALL: all {len(rows)} cells learned -- both networks learn from real "
          "images on every device and arm tested, on every cached subset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
