# E001 — The documented `trainer` CLI invocation is broken

| | |
|---|---|
| **Status** | REPRODUCED (defect still open) |
| **Cards** | `P1-3` |
| **Command** | `python trainer train GoogLeNet --dry-run` |
| **Host** | Windows host, Python as installed on the host |
| **First recorded** | [HANDOFF_001 §3a](../handoffs/HANDOFF_001.md) |

## Claim

Both forms documented in `README.md` and `trainer/README.md` crash. Only `python -m trainer`
works.

## Output

```
$ python trainer train GoogLeNet --dry-run
  File "...\trainer\__main__.py", line 2, in <module>
    from trainer import main
  File "...\trainer\trainer.py", line 7, in <module>
    from trainer.model_base import Model, load_model_class
ModuleNotFoundError: No module named 'trainer.model_base'; 'trainer' is not a package

$ python trainer/trainer.py train GoogLeNet --dry-run
  File "...\trainer\trainer.py", line 7, in <module>
    from trainer.model_base import Model, load_model_class
ModuleNotFoundError: No module named 'trainer.model_base'; 'trainer' is not a package
```

## Root cause

Run as a script, `trainer/` becomes `sys.path[0]`, so `import trainer` resolves to the folder
instead of the package and `from trainer.model_base import ...` at `trainer/trainer.py:7` fails.
The `try/except ModuleNotFoundError` fallbacks present in the `trainer_model.py` files were never
applied to `trainer.py` itself.

## Working form (use this until `P1-3` lands)

```
$ python -m trainer register --model-name GoogLeNet --model-dir GoogLeNet --force
Registered model 'GoogLeNet' from ...\GoogLeNet\trainer_model.py
Default train parameters:
  - version_name: googlenet_v1
  - num_classes: 64
  - checkpoint_path: /workspace/GoogLeNet/checkpoints
  - train_samples: 51200
  - val_samples: 2048
  - batch_size: 64

$ python -m trainer train GoogLeNet --dry-run
Executing training command:
...\python.exe ...\GoogLeNet\train.py --version_name googlenet_v1 --num_classes 64
  --checkpoint_path /workspace/GoogLeNet/checkpoints --train_samples 51200
  --val_samples 2048 --batch_size 64
```

The generated `trainer/registry.json` was deleted afterwards (it is gitignored anyway).
