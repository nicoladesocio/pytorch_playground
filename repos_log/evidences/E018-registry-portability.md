# E018 — Registry stores paths relative to the repo root

| | |
|---|---|
| **Status** | CONFIRMED → fixed in HANDOFF_003 |
| **Cards** | `P3-5` |
| **Command** | `python probes/check_registry.py` |
| **First recorded** | [HANDOFF_003 §6.7](../handoffs/HANDOFF_003.md) |

## Output

```
[P3-5] OK   alexnet.model_dir relative: AlexNet
[P3-5] OK   alexnet.module_file relative: AlexNet/trainer_model.py
[P3-5] OK   googlenet.model_dir relative: GoogLeNet
[P3-5] OK   googlenet.module_file relative: GoogLeNet/trainer_model.py
[P3-5] dry-run alexnet:   rc=0  ... /workspace/AlexNet/train.py --version_name alexnet_v1 ...
[P3-5] dry-run googlenet: rc=0  ... /workspace/GoogLeNet/train.py --version_name googlenet_v1 ...
[P3-5] PASS
```

## Why it mattered

`registry.json` stored **absolute** `model_dir`/`module_file` paths, so a registry generated in the
container (`/workspace/...`) was invalid on the host and vice versa. Paths are now relative to
`PROJECT_ROOT`; `_abs_from_root()` still accepts a legacy absolute entry, so an existing
`registry.json` keeps working.

Note `registry.json` is gitignored, so every fresh checkout must re-register — still open.
