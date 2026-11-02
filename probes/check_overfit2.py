"""Overfit-one-batch test for both networks (CPU, small image).

If a network cannot drive a single batch's loss well below chance, its
architecture or initialisation is broken. This isolates that from data volume.
"""
import sys
import time

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from AlexNet.model import AlexNet
from GoogLeNet.model import GoogLeNet

K = 10
B = 32
IMG = 96
STEPS = 120
LR = 1e-3

torch.manual_seed(0)
x = torch.randn(B, 3, IMG, IMG)
y = torch.randint(0, K, (B,))
chance = float(torch.log(torch.tensor(float(K))))


def run(name, make):
    net = make()
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rows = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        if name == "GoogLeNet":
            out, aux2, aux1 = net(x)
            loss = (
                F.cross_entropy(out, y)
                + 0.3 * F.cross_entropy(aux1, y)
                + 0.3 * F.cross_entropy(aux2, y)
            )
        else:
            out = net(x)
            loss = F.cross_entropy(out, y)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item()
        opt.step()
        if step in (1, 5, 10, 20, 40, 60, 80, 100, STEPS):
            acc = (out.argmax(1) == y).float().mean().item()
            rows.append((step, loss.item(), grad, acc))
    dt = time.time() - t0
    print(f"\n{name}: chance loss = {chance:.4f}, {STEPS} steps in {dt:.0f}s")
    print("  step      loss   grad_norm  train_acc")
    for step, loss, grad, acc in rows:
        print(f"  {step:4d}  {loss:8.4f}  {grad:9.3e}  {acc:8.3f}")
    first, last = rows[0][1], rows[-1][1]
    verdict = "LEARNED" if last < chance * 0.6 else "FAILED to learn"
    print(f"  -> {first:.4f} -> {last:.4f} ({100*(first-last)/first:+.1f}%) : {verdict}")


run("AlexNet", lambda: AlexNet(num_classes=K, image_size=IMG))
run("GoogLeNet", lambda: GoogLeNet(num_classes=K, image_size=IMG))
