"""Why does AlexNet's SGD arm refuse to learn, and why does Adam not mind?

Kanban `P2-25` records that `gradient_clip_val=1.0` scales *every* SGD step by 0.037
because AlexNet's **total** gradient norm at initialisation is 26.9. That explains why
the clipped arm is slow. It does not explain the other half of the observation: with the
clip disabled, `lr=0.1` diverges to NaN within a handful of steps while `lr=0.01` sits
exactly on chance for a whole epoch. A single global learning rate cannot do both.

A total norm is the wrong statistic for that question. `torch.optim.SGD` applies the same
`lr` to every parameter, so what matters is the *per-parameter* gradient scale: the layer
whose gradient is 1000x larger moves 1000x faster in weight space, and it is that layer
which either explodes or stalls. This probe prints the per-layer norms so the spread is
visible instead of inferred.

The mechanism this repo is dealing with is the initialisation, not the optimiser.
`AlexNet/model.py` sets `normal_(std=0.01)` on **every** conv and **every** Linear.
`std=0.01` is a fixed number, but each layer's fan-in is different, so the forward
variance multiplier `fan_in * std**2` differs by orders of magnitude between conv1
(fan_in 363 -> 0.036, activations shrink ~5x) and fc1 (fan_in 9216 -> 0.92). Gradients
back-propagated through that stack inherit the imbalance.

    python probes/check_grad_scale.py                      # both nets, cuda if present
    python probes/check_grad_scale.py --model AlexNet --device cuda
    python probes/check_grad_scale.py --batch 32 --image-size 227 --precision 16-mixed

Exit code is 0 if the measurement completed; this probe reports, it does not assert.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

REPO_ROOT = _bootstrap.REPO_ROOT
sys.path.insert(0, os.path.join(REPO_ROOT, "probes"))

# GoogLeNet's combined loss is main + 0.3*aux1 + 0.3*aux2, so its chance level is
# 1.6*ln(K), not ln(K) (kanban P2-17, evidence E019). Kept local so the probe can run
# without importing check_overfit2's argparse-time side effects.
AUX_WEIGHT = 0.3


def load_model_module(dir_name, module_name):
    """Import AlexNet/model.py or GoogLeNet/model.py under a unique module name.

    Both files are named ``model.py`` and both are imported as bare ``import model``, so
    importing them both normally makes the second silently reuse the first -- see
    AGENT_INSTRUCTIONS.md section 3.5. Same pattern as probes/check_hardware.py.
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


def fan_in(param, kind):
    """Number of inputs feeding one output unit, for the variance-multiplier column."""
    if kind == "conv":
        return int(param.shape[1] * torch.tensor(param.shape[2:]).prod().item())
    return int(param.shape[1])


def describe(name, param):
    """Classify a parameter and report its init std, so the init can be audited."""
    if param.dim() == 4:
        kind = "conv"
    elif param.dim() == 2:
        kind = "linear"
    elif param.dim() == 1:
        kind = "bias/ln"
    else:
        kind = "other"
    std = float(param.detach().float().std()) if param.numel() > 1 else float("nan")
    return kind, std


def measure(name, mod, device, batch, image_size, classes, precision, seed):
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed_all(seed)

    net = mod.__dict__[name](num_classes=classes, image_size=image_size)
    dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float32
    net = net.to(device)

    # A fixed random batch of the same shape the training run uses. Real images are not
    # needed here: the question is how the *architecture and its init* distribute
    # gradient magnitude, and a Gaussian input at the right scale (ImageNet-normalised)
    # reproduces the imbalance without a 5.8 GB cache dependency.
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    y = torch.randint(0, classes, (batch,), device=device)

    rows = []
    with torch.autocast(device_type="cuda" if device != "cpu" else "cpu",
                        dtype=dtype, enabled=precision != "32"):
        if name == "GoogLeNet":
            out, aux2, aux1 = net(x)
            loss = (F.cross_entropy(out, y)
                    + AUX_WEIGHT * F.cross_entropy(aux1, y)
                    + AUX_WEIGHT * F.cross_entropy(aux2, y))
        else:
            out = net(x)
            loss = F.cross_entropy(out, y)
    loss.backward()

    for pname, param in net.named_parameters():
        if param.grad is None:
            continue
        kind, std = describe(pname, param)
        g = float(param.grad.detach().float().norm(2))
        w = float(param.detach().float().norm(2))
        rows.append({
            "name": pname, "kind": kind, "fan_in": (fan_in(param.detach(), kind)
                                                    if kind in ("conv", "linear") else None),
            "init_std": std, "grad_norm": g, "weight_norm": w,
            # The quantity SGD actually cares about: the weight-space step size this
            # layer takes at lr=1. A layer 1000x above the median needs lr 1000x smaller.
            "step_at_lr1": g,
        })
    total = sum(r["grad_norm"] ** 2 for r in rows) ** 0.5
    return loss.item(), total, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="both", choices=["AlexNet", "GoogLeNet", "both"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=227)
    ap.add_argument("--classes", type=int, default=64)
    ap.add_argument("--precision", default="32", choices=["32", "16-mixed", "bf16-mixed"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but CUDA is unavailable. "
                 "Run probes/check_hardware.py.")

    print(f"torch {torch.__version__}  device={device}  batch={args.batch}  "
          f"image_size={args.image_size}  classes={args.classes}  "
          f"precision={args.precision}  seed={args.seed}")
    print("Interpretation: SGD multiplies every gradient by the SAME lr, so the column "
          "that decides whether a layer learns is grad_norm relative to the median. "
          "A spread of 10^3+ across layers means no single lr can train the network.")

    models = ["AlexNet", "GoogLeNet"] if args.model == "both" else [args.model]
    mods = {"AlexNet": ("AlexNet", "ppg_alexnet_model"),
            "GoogLeNet": ("GoogLeNet", "ppg_googlenet_model")}

    for name in models:
        dir_name, mod_name = mods[name]
        mod = load_model_module(dir_name, mod_name)
        t0 = time.time()
        loss, total, rows = measure(name, mod, device, args.batch, args.image_size,
                                    args.classes, args.precision, args.seed)
        norms = sorted(r["grad_norm"] for r in rows)
        med = norms[len(norms) // 2]
        print(f"\n--- {name}: loss {loss:.4f}  TOTAL grad norm {total:.4f}  "
              f"({len(rows)} tensors, {time.time() - t0:.1f}s) ---")
        print(f"  median per-tensor grad norm {med:.3e}   "
              f"min {norms[0]:.3e}   max {norms[-1]:.3e}   "
              f"spread {norms[-1] / max(norms[0], 1e-30):.3e}")
        print(f"  => a clip at 1.0 scales the total by {min(1.0, 1.0 / total):.4f}; "
              f"lr=0.1 therefore behaves as lr={0.1 * min(1.0, 1.0 / total):.5f}")
        print(f"\n  {'parameter':<34}{'kind':<9}{'fan_in':>8}{'init_std':>11}"
              f"{'grad_norm':>13}{'x median':>11}")
        print("  " + "-" * 86)
        for r in sorted(rows, key=lambda r: -r["grad_norm"]):
            fi = f"{r['fan_in']}" if r["fan_in"] else "-"
            print(f"  {r['name']:<34}{r['kind']:<9}{fi:>8}{r['init_std']:>11.4f}"
                  f"{r['grad_norm']:>13.3e}{r['grad_norm'] / max(med, 1e-30):>11.2e}")

        # Group the way an SGD-vs-Adam argument actually needs: conv stack vs classifier.
        groups = {}
        for r in rows:
            key = "features" if r["name"].startswith("features") else "classifier"
            groups.setdefault(key, []).append(r["grad_norm"])
        print("\n  group summary (this is the P2-25 mechanism):")
        for key, vals in groups.items():
            print(f"    {key:<12} n={len(vals):<4} "
                  f"max {max(vals):.3e}  median {sorted(vals)[len(vals) // 2]:.3e}")
        if len(groups) == 2:
            a = max(groups["classifier"])
            b = max(groups["features"])
            print(f"    classifier/features max-norm ratio: {a / max(b, 1e-30):.3e}")
            print("    If that ratio is large, one global lr cannot serve both groups: "
                  "the fast group diverges at the lr the slow group needs. That is a "
                  "property of the uniform std=0.01 init, not of SGD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
