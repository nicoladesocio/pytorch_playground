# E009 — AlexNet's hardcoded `256*6*6` flatten size is a latent crash

| | |
|---|---|
| **Status** | REPRODUCED → fixed in HANDOFF_003 |
| **Cards** | `P2-13` |
| **Command** | construct `AlexNet(...)` at several `image_size` values |
| **First recorded** | [HANDOFF_002 §4](../handoffs/HANDOFF_002.md) |

## Measured

`AlexNet/model.py:39` used `nn.Linear(256 * 6 * 6, 4096)`, but conv1 is `stride=4` with **no
padding**, so spatial size is `floor((H-11)/4)+1` then pooled.

| `image_size` | flatten features | result |
|---|---|---|
| 224 | 6400 | **RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x6400 and 9216x9216)** |
| 227 | 9216 | OK |
| 231 | 9216 | OK |
| 256 | 9216 | OK |

It worked *only* because both `train.py` files happened to pass `image_size=227`, while
`ImageNetSubsetDataset`'s own default is `image_size=224`. Any caller using the default crashed.

## Resolution

Flatten size derived from the model's own conv stack (`_flatten_features()`); a bad size raises a
clear message naming the expected size (HANDOFF_003).
