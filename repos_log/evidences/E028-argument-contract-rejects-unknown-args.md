# E028 — `train_final.py` now rejects unknown arguments

| | |
|---|---|
| **Status** | CONFIRMED — fixed and verified |
| **Cards** | `P2-24` |
| **Command** | `python AlexNet/train_final.py --bogus` |
| **Host** | `ppg-dev` |
| **First recorded** | [HANDOFF_004 §3.15](../handoffs/HANDOFF_004.md) |

## The defect

Both scripts tested their single flag with membership rather than parsing:

```python
smoke = "--smoke" in sys.argv[1:]
```

Anything unrecognised is simply not `"--smoke"`, so it is discarded and the script proceeds with
the **full** `CONFIG`. The comment above the line claimed `--smoke` was "intentionally the only
accepted argument" — true of the intent, false of the code.

Observed before the fix: `python AlexNet/train_final.py --bogus` exited **0** and printed the
full-run banner (`Chance-level accuracy for 64 classes is 1.56%`), i.e. it had started a real
training run. A mistyped `--smok` would not fail; it would silently run the multi-hour experiment
and still print a RESULT block.

## After the fix

```
$ python AlexNet/train_final.py --bogus
AlexNet/train_final.py: unrecognised argument(s): --bogus
AlexNet/train_final.py accepts no arguments except: --smoke.
Hyperparameters live in the CONFIG dict at the top of the file; use `trainer train <model> --set key=value` when you need to override one.
exit=1

$ python GoogLeNet/train_final.py --lr 0.1
GoogLeNet/train_final.py: unrecognised argument(s): --lr, 0.1
...
exit=1
```

## Unit-level checks

```
no args -> False
--smoke -> True
rejected ['--bogus']            -> AlexNet/train_final.py: unrecognised argument(s): --bogus
rejected ['--smok']             -> AlexNet/train_final.py: unrecognised argument(s): --smok
rejected ['--smoke','--lr','0.1'] -> AlexNet/train_final.py: unrecognised argument(s): --lr, 0.1
```

## No regression

`--smoke` still runs and its RESULT block is byte-identical to
[E021](./E021-gpu-smoke-runs.md); the no-argument path still starts the full run.

## Implementation note

`utils/runtime.validate_final_args()` is shared by both scripts so they cannot drift apart. It
raises `SystemExit` rather than using `argparse` deliberately: these must stay parameterless
deliverables, and `argparse` would advertise a flag surface that does not exist.
