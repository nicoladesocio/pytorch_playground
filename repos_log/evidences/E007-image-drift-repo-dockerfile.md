# E007 — The image in use was not built from the repo `Dockerfile`

| | |
|---|---|
| **Status** | CONFIRMED (open as `P2-12`; worked around by `Dockerfile.gpu`) |
| **Cards** | `P2-12`, `P1-5`, `P2-20` |
| **Command** | `docker image inspect nicoladesocio/pytorchplayground:latest` |
| **First recorded** | [HANDOFF_002 §3](../handoffs/HANDOFF_002.md) |

## The drift

`nicoladesocio/pytorchplayground:latest` (33.5 GB) exists locally but was **not** built from the
`Dockerfile` in this repo:

| | Repo `Dockerfile` | Image on disk |
|---|---|---|
| Base | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` | PyTorch **2.13.0** + CUDA **13.2.1** |
| Python | (base default) | 3.12.3 |
| `code-server` | no | **yes**, as `ENTRYPOINT` |
| `nscacert.crt` | required (build fails without it) | corporate-cert layers absent |

## Practical consequences

- `docker run <image> python3 ...` **fails** — you must pass `--entrypoint`.
- The image is **missing most declared dependencies**. Install once per container:
  `pip install --break-system-packages lightning datasets tensorboard scipy pandas scikit-learn tqdm`.
- The repo source is **bind-mounted**, so you never need to rebuild the image to iterate on
  Python code. This is why the drift has not bitten yet — and why it is easy to miss.

Note that `pip install .` of a hatchling project whose `packages` dirs are absent still
**succeeds silently**, so the dependency layer can pass while installing nothing.

## Resolution

HANDOFF_004 added `Dockerfile.gpu`: no cert, no `code-server`, no vendor registry, and an import
gate for every declared dependency so a missing dep fails the **build** instead of the run.
