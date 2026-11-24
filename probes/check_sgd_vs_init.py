"""Is AlexNet's SGD failure an optimiser verdict or an initialisation verdict?

`P2-25` asked whether `gradient_clip_val=1.0` is what pins AlexNet's SGD arm at chance.
The grid in `evidences/E033` says no: with the clip disabled and `lr` swept over two
decades (1e-3 .. 1e-2) SGD still sits on chance, and `lr=0.1` diverges to NaN -- while
Adam at the *same* budget reaches 4.6x chance. So the clip was never the whole story.

`probes/check_grad_scale.py` supplies the candidate mechanism: `AlexNet/model.py` applies
`normal_(std=0.01)` to **every** conv and **every** Linear. `std` is a fixed number but
each layer's fan-in is not, so the forward variance multiplier `fan_in * std**2` ranges
from 0.036 (conv1, fan_in 363) to 0.92 (fc1, fan_in 9216). Measured per-tensor gradient
norms at init span **1.5e4** (conv1.bias 1.4e-3 -> fc3 2.1e+1). SGD multiplies every one
of those by the *same* lr, so the lr that lets the slow layers move is far past the lr
that keeps the fast layers finite. Adam divides by each parameter's own gradient RMS, so
it is immune to exactly this spread -- which is the whole difference between the two arms.

This probe separates the two explanations. It trains the real `AlexNet` on real cached
images with unclipped SGD under three initialisations:

  paper   -- the repo's current uniform `normal_(std=0.01)` (the control)
  kaiming -- `kaiming_normal_(mode='fan_in', nonlinearity='relu')` on conv+Linear,
             biases zeroed except a small positive constant where the paper wants live ReLUs
  torch   -- PyTorch's own defaults (what you get if you delete the init block)

Each init is run on `--batches` INDEPENDENT fixed batches (all inits share the same
batches, so arms differ ONLY in init). Memorising a single 32-image batch is easy enough
that which 32 you draw decides a PASS/FAIL, so a one-batch run is not evidence -- the
probe reports how many of the batches each init memorises.

If `paper` memorises every batch, the uniform init is NOT what blocks SGD (the blocker is
elsewhere: the generalisation gap, the regularisers, or the data). If `paper` memorises
none while `torch`/`kaiming` do, the init is the blocker. If none memorise any batch, the
optimiser/lr/budget is the blocker, not the init -- check the Adam control first.

    python probes/check_sgd_vs_init.py --device cuda --steps 600 --lr 0.01 --batches 4
    python probes/check_sgd_vs_init.py --init kaiming --lr 0.05 --steps 1200
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import _bootstrap  # noqa: F401

REPO_ROOT = _bootstrap.REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, "probes"))

from check_learn import build_batch, find_caches, load_model_module, make_arm_transform  # noqa: E402


def reinit(net, mode):
    """Replace AlexNet's initialisation in place. Returns a human-readable summary."""
    lines = []
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            if mode == "paper":
                # Exactly what AlexNet/model.py does today.
                nn.init.normal_(mod.weight, mean=0, std=0.01)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
            elif mode == "kaiming":
                nn.init.kaiming_normal_(mod.weight, mode="fan_in", nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)
            elif mode == "torch":
                # Reset to the framework default for this layer type.
                mod.reset_parameters()
            lines.append(f"{name}:{mod.__class__.__name__}")
    # Keep the paper's bias=1 trick on the ReLU-fed hidden layers so the *only*
    # difference between arms is the weight init, not dead-ReLU behaviour.
    if mode == "kaiming":
        for i, mod in enumerate([m for m in net.classifier if isinstance(m, nn.Linear)]):
            if i != len([m for m in net.classifier if isinstance(m, nn.Linear)]) - 1:
                nn.init.constant_(mod.bias, 1.0)
        convs = [m for m in net.features if isinstance(m, nn.Conv2d)]
        for idx in (1, 3, 4):
            if idx < len(convs) and convs[idx].bias is not None:
                nn.init.constant_(convs[idx].bias, 1.0)
    return lines


def grad_spread(net):
    """Per-tensor gradient-norm spread at init -- the quantity SGD cannot cope with."""
    norms = []
    for _, p in net.named_parameters():
        if p.grad is not None:
            norms.append(float(p.grad.detach().float().norm(2)))
    norms = [n for n in norms if n > 0]
    if not norms:
        return float("nan"), float("nan"), float("nan")
    return min(norms), sorted(norms)[len(norms) // 2], max(norms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    # "51200" matches BOTH the 2048-val and the 16384-val caches, and find_caches()
    # sorts by mtime -- so the cache actually used silently followed whichever was
    # built last. Pin the canonical 2048-val cache (the one E031/E033 use) by default.
    ap.add_argument("--cache", default="51200t_2048v")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batches", type=int, default=4,
                    help="independent fixed batches each init must memorise; a single "
                         "batch cannot decide this question (see the note in main())")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=227)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--init", default="all", choices=["all", "paper", "kaiming", "torch"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but unavailable; run probes/check_hardware.py")

    caches = find_caches(only=args.cache)
    if not caches:
        sys.exit(f"No cache matching {args.cache!r}. Build one with probes/check_big.py")
    cache = caches[0]

    alex_mod = load_model_module("AlexNet", "ppg_alexnet_model")

    # Build --batches INDEPENDENT fixed batches. Every init sees the SAME set of
    # batches, so the arms still differ ONLY in initialisation -- but the verdict is
    # no longer decided by one lucky draw.
    #
    # Why this matters: with a single batch, this probe's own verdict flipped between
    # caches. On the 2048-val cache the paper init memorised (loss 2.3025 -> 0.0003,
    # acc 1.000); on the 16384-val cache, at the same lr and step count, it stalled at
    # 2.0809 / acc 0.250. Memorising 32 images is easy enough that which 32 you draw
    # decides the answer, and a probe whose verdict moves with the batch cannot close
    # a card (AGENT_INSTRUCTIONS 4.2, E025). Report the fraction of batches each init
    # memorises instead.
    batches = []
    for bi in range(args.batches):
        pils, y, K, target_classes = build_batch(cache, args.classes, args.batch,
                                                 args.image_size, args.seed + bi)
        transform = make_arm_transform(cache, target_classes, args.image_size, "eval")
        x = torch.stack([transform(p) for p in pils]).to(device)
        batches.append((x, y.to(device), K))

    K = batches[0][2]
    # Chance for a K-class cross-entropy is ln(K); the pass rule matches check_learn
    # (loss below 0.6x chance) so the two probes agree on what "learned" means.
    chance = math.log(K)
    print(f"torch {torch.__version__}  device={device}  cache={os.path.basename(cache)}")
    print(f"batch={args.batch} classes={K} image_size={args.image_size} "
          f"optimizer={args.optimizer} lr={args.lr} steps={args.steps} seed={args.seed} "
          f"batches={args.batches}")
    print(f"chance loss ln({K}) = {chance:.4f}; PASS rule = final loss < {0.6 * chance:.4f}")
    print(f"(each init is run on all {args.batches} shared batches; arms differ ONLY in --init)\n")

    inits = ["paper", "kaiming", "torch"] if args.init == "all" else [args.init]
    rows = []
    for mode in inits:
        per_batch = []
        lo = med = hi = float("nan")
        t0 = time.time()
        for bi, (x, yc, _) in enumerate(batches):
            # Re-seed per batch so the WEIGHTS are identical across batches: the only
            # thing that varies is which 32 images are being memorised.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            net = alex_mod.AlexNet(num_classes=K, image_size=args.image_size).to(device)
            reinit(net, mode)
            net.train()

            if bi == 0:
                # Measure the gradient spread this init produces, before any update.
                for p in net.parameters():
                    p.grad = None
                F.cross_entropy(net(x), yc).backward()
                lo, med, hi = grad_spread(net)
                for p in net.parameters():
                    p.grad = None

            if args.optimizer == "sgd":
                opt = torch.optim.SGD(net.parameters(), lr=args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            else:
                opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)

            first = last = None
            nan_at = None
            out = None
            for step in range(1, args.steps + 1):
                opt.zero_grad(set_to_none=True)
                out = net(x)
                loss = F.cross_entropy(out, yc)
                if not torch.isfinite(loss):
                    nan_at = step
                    break
                loss.backward()
                # Deliberately NO clipping: P2-25 established the clip is not the blocker.
                opt.step()
                if first is None:
                    first = loss.item()
                last = loss.item()
            acc = (out.argmax(1) == yc).float().mean().item()
            status = ("NaN" if nan_at else
                      ("PASS" if last is not None and last < chance * 0.6 else "FAIL"))
            per_batch.append((first, last, acc, nan_at, status))

        n_pass = sum(1 for r in per_batch if r[4] == "PASS")
        rows.append((mode, per_batch, lo, med, hi, n_pass))
        print(f"  init={mode:<8} grad-norm spread: min {lo:.3e} med {med:.3e} "
              f"max {hi:.3e}  ratio {hi / max(lo, 1e-30):.3e}")
        for bi, (first, last, acc, nan_at, status) in enumerate(per_batch):
            print(f"           batch {bi} (seed {args.seed + bi}): loss {first:.4f} -> "
                  f"{'NaN@step %d' % nan_at if nan_at else '%.4f' % last}   "
                  f"acc {acc:.3f}   {status}")
        print(f"           memorised {n_pass}/{len(per_batch)} batches   "
              f"{time.time() - t0:.1f}s\n")

    print("=" * 78)
    print(f"{'init':<10}{'memorised':>11}{'median last':>13}{'mean acc':>10}"
          f"{'grad spread':>16}  result")
    print("-" * 78)
    for mode, per_batch, lo, med, hi, n_pass in rows:
        lasts = sorted(r[1] for r in per_batch if r[1] is not None)
        med_last = lasts[len(lasts) // 2] if lasts else float("nan")
        mean_acc = sum(r[2] for r in per_batch) / len(per_batch)
        status = ("PASS" if n_pass == len(per_batch) else
                  ("PARTIAL" if n_pass else "FAIL"))
        print(f"{mode:<10}{n_pass:>6}/{len(per_batch)}{med_last:>13.4f}{mean_acc:>10.3f}"
              f"{(hi / max(lo, 1e-30)):>16.3e}  {status}")
    print("=" * 78)
    all_pass = [r[0] for r in rows if r[5] == len(batches)]
    some_pass = [r[0] for r in rows if 0 < r[5] < len(batches)]
    if all_pass and "paper" in all_pass:
        print("The paper's std=0.01 init memorises every batch, so the uniform init is NOT "
              "what blocks SGD.")
    elif "paper" in some_pass:
        print("The paper's std=0.01 init memorises only SOME batches. A single-batch run of "
              "this probe\nis therefore not evidence either way -- raise --batches before "
              "drawing a conclusion.")
    elif all_pass:
        print("Inits that memorise every batch: " + ", ".join(all_pass) +
              ". If 'paper' is absent while others are present, the uniform std=0.01 init "
              "is the blocker.")
    else:
        print("No initialisation memorised any batch with this optimizer/lr/budget. Either "
              "the optimizer\ncannot train this stack at this lr, or --steps is too short -- "
              "check the Adam control\nbefore concluding anything about the init.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
