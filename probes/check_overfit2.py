"""Overfit-one-batch test for both networks, on whatever device you have.

If a network cannot drive a single batch's loss well below chance, its architecture or
initialisation is broken. This isolates that question from data volume, augmentation and
learning rate, which is why it is the first thing to run when a loss looks flat.

    python probes/check_overfit2.py                      # best device available
    python probes/check_overfit2.py --device cpu         # reproduce HANDOFF_003 §6.3
    python probes/check_overfit2.py --device cuda --steps 200

``--image_size`` defaults to **160**, not 96. GoogLeNet's ``InceptionOutClassifier`` uses
``AvgPool2d(kernel_size=5, stride=3)``, which spatially collapses below ~128 px: at 96 px
the aux heads see a single spatial position and stop acting as a deep-supervision branch
(HANDOFF_003 §3.9, kanban `P2-16`). Judging GoogLeNet's ability to learn at 96 px therefore
measures a crippled branch. HANDOFF_003 §6.3's numbers were taken at 96 px, so pass
``--image_size 96`` to reproduce them.

The chance threshold differs per network and is computed, not assumed (kanban `P2-17`):
AlexNet's loss is ``ln(K)``, but GoogLeNet trains on ``main + 0.3*aux1 + 0.3*aux2``, so with
all three heads uniform at init its expected *starting* loss is ``1.6*ln(K)``. Comparing
GoogLeNet's combined loss to ``ln(K)`` reports a healthy, untrained network as broken — a
mistake that cost real debugging time in HANDOFF_003.
"""
import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from AlexNet.model import AlexNet
from GoogLeNet.model import GoogLeNet

# Combined-loss weight GoogLeNet's training step applies to each aux head.
AUX_WEIGHT = 0.3


def chance_loss(name, k):
    """Expected loss of an untrained network, per network (kanban `P2-17`).

    Shared with any other probe that has to answer "did this beat chance?".
    """
    if name == "GoogLeNet":
        return (1.0 + 2.0 * AUX_WEIGHT) * math.log(k)
    return math.log(k)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto picks cuda when available")
    ap.add_argument("--image_size", type=int, default=160,
                    help="160 keeps GoogLeNet's aux heads usable; use 96 to reproduce "
                         "HANDOFF_003 section 6.3")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is False. "
                 "Run probes/check_hardware.py to find out why.")

    if args.image_size < 128:
        print(f"WARNING: image_size={args.image_size} collapses GoogLeNet's aux heads "
              "(P2-16); its verdict is not a fair test of the architecture.")

    K, B, IMG = args.classes, args.batch, args.image_size
    torch.manual_seed(0)
    x = torch.randn(B, 3, IMG, IMG, device=device)
    y = torch.randint(0, K, (B,), device=device)

    dev = f"{device}"
    if device == "cuda":
        dev += f" ({torch.cuda.get_device_name(0)})"
    print(f"device={dev}  B={B} K={K} img={IMG} steps={args.steps} lr={args.lr}")

    results = {}
    for name, make in (("AlexNet", lambda: AlexNet(num_classes=K, image_size=IMG)),
                       ("GoogLeNet", lambda: GoogLeNet(num_classes=K, image_size=IMG))):
        results[name] = run(name, make, x, y, K, args)

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\nOVERALL: FAILED -> {', '.join(failed)}")
        return 1
    print("\nOVERALL: both networks learned (loss < 0.6 x their own chance level)")
    return 0


def run(name, make, x, y, K, args):
    device = x.device
    net = make().to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    chance = chance_loss(name, K)
    rows = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        if name == "GoogLeNet":
            out, aux2, aux1 = net(x)
            loss = (
                F.cross_entropy(out, y)
                + AUX_WEIGHT * F.cross_entropy(aux1, y)
                + AUX_WEIGHT * F.cross_entropy(aux2, y)
            )
        else:
            out = net(x)
            loss = F.cross_entropy(out, y)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item()
        opt.step()
        if step in (1, 5, 10, 20, 40, 60, 80, 100, args.steps):
            acc = (out.argmax(1) == y).float().mean().item()
            rows.append((step, loss.item(), grad, acc))
    dt = time.time() - t0
    print(f"\n{name}: chance loss = {chance:.4f}, {args.steps} steps in {dt:.0f}s")
    print("  step      loss   grad_norm  train_acc")
    for step, loss, grad, acc in rows:
        print(f"  {step:4d}  {loss:8.4f}  {grad:9.3e}  {acc:8.3f}")
    first, last = rows[0][1], rows[-1][1]
    # 0.6 x chance is the bar HANDOFF_003 used; it separates "memorising the batch" from
    # "drifting". Compared against the network's OWN chance level, per P2-17.
    ok = last < chance * 0.6
    print(f"  -> {first:.4f} -> {last:.4f} ({100*(first-last)/first:+.1f}%) : "
          f"{'LEARNED' if ok else 'FAILED to learn'}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
