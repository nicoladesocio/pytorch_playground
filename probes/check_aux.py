"""Report GoogLeNet aux-head spatial sizes and per-head losses vs chance.

At small --image_size the aux head's AvgPool2d(kernel=5, stride=3) can collapse
to 1x1 (or 0), which cripples the auxiliary classifier and slows the whole
network. This prints the real geometry per input size, plus main/aux losses
separately so the chance level of the COMBINED loss is unambiguous.
"""
import sys
import time

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

from GoogLeNet.model import GoogLeNet

K = 10
B = 16
W = 0.3
CHANCE = float(torch.log(torch.tensor(float(K))))
EXPECTED_TOTAL_CHANCE = (1 + 2 * W) * CHANCE

for img in (96, 128, 160, 224):
    net = GoogLeNet(num_classes=K, image_size=img)
    s4a, s4d = net._aux_spatial_sizes(img)
    print(f"\nimage_size={img}: aux tap spatial 4a={s4a} 4d={s4d} | "
          f"aux1 feat_map={net.aux1.feature_map_size} "
          f"aux2 feat_map={net.aux2.feature_map_size}")
    if net.aux1.feature_map_size < 2 or net.aux2.feature_map_size < 2:
        print("  !! aux head spatially collapsed -> aux classifier is crippled")

    torch.manual_seed(0)
    x = torch.randn(B, 3, img, img)
    y = torch.randint(0, K, (B,))
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(1, 41):
        opt.zero_grad(set_to_none=True)
        out, aux2, aux1 = net(x)
        lm = F.cross_entropy(out, y)
        l1 = F.cross_entropy(aux1, y)
        l2 = F.cross_entropy(aux2, y)
        loss = lm + W * l1 + W * l2
        loss.backward()
        opt.step()
        if step in (1, 2, 5, 10, 20, 40):
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"  step {step:3d} main={lm.item():6.3f} aux1={l1.item():6.3f} "
                  f"aux2={l2.item():6.3f} total={loss.item():6.3f} acc={acc:.3f}")
    print(f"  chance: main={CHANCE:.3f} total={EXPECTED_TOTAL_CHANCE:.3f} "
          f"({40} steps in {time.time()-t0:.0f}s)")
