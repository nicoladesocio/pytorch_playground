# Trainer CLI: Register + Train

This folder now contains a generic training CLI that supports both:

- Registering a model from a sibling model folder
- Training a previously registered model with defaults + overrides

## What Was Implemented

### 1. Subcommand-based CLI

Implemented in `trainer/trainer.py` with two commands:

- `register`
- `train`

Examples:

```bash
python3 trainer register --model-name GoogLeNet --model-version GoogLeNet
python3 trainer train GoogLeNet
```

### 2. Persistent registry

A JSON registry file is stored at:

- `trainer/registry.json`

Each entry contains:

- model name
- list(model_versions)

a model_version contains:
- version name
- version description
- `trainer_model.py` path
- model class name
- default parameters
- train script name

### 3. Base model abstraction

Implemented in `trainer/model_base.py`:

- `class Model`
- Common defaults and parameter resolution
- Folder-level parameter loading from `trainer_params.json`
- Conversion of resolved parameters to CLI arguments
- Train command builder

This enables inheritance per model folder.

### 4. Model-specific subclasses in sibling folders

Added:

- `GoogLeNet/trainer_model.py`
- `AlexNet/trainer_model.py`

Each class inherits from `Model` and defines:

- `name`
- `train_script`
- `default_parameters` (including `version_name`)

### 5. Train scripts accept `version_name`

Updated:

- `GoogLeNet/train.py`
- `AlexNet/train.py`

Both now accept `--version_name` and use it for TensorBoard logger name.

### 6. Invocation compatibility

Code was adjusted to support both:

```bash
python3 trainer/trainer.py <subcommand> ...
python3 trainer <subcommand> ...
```

(Import fallbacks were added for script/package execution contexts.)

## Parameter Resolution Order

For `python3 trainer train <model>` parameters are resolved in this order (later overrides earlier):

1. `default_parameters` in model subclass (`trainer_model.py`)
2. Optional model-folder file: `trainer_params.json`
3. Explicit train CLI options (example: `--batch-size 128`)
4. Generic overrides via `--set KEY=VALUE`

## CLI Usage

### Register

```bash
python3 trainer register --model-name GoogLeNet --model-dir GoogLeNet
python3 trainer register --model-name AlexNet --model-dir AlexNet
```

Optional flags:

- `--force` overwrite an existing registration
- `--model-class` specify subclass name if more than one exists

### Train

```bash
python3 trainer train GoogLeNet
python3 trainer train GoogLeNet --version-name googlenet_exp_01 --batch-size 128
python3 trainer train GoogLeNet --set max_epochs=20 --set custom_flag=true
python3 trainer train GoogLeNet --dry-run
```

Notes:

- `--dry-run` prints the resolved command without running training.
- Unknown extra arguments are passed through to the model train script.

## Expected Model Folder Contract

Each model folder should contain:

1. `train.py`
2. `trainer_model.py` with one subclass of `Model`
3. Optional `trainer_params.json` for folder-local defaults

Example `trainer_params.json`:

```json
{
  "version_name": "googlenet_debug_v2",
  "batch_size": 96,
  "train_samples": 10240
}
```

## Next Steps

1. Register both models:

```bash
python3 trainer register --model-name GoogLeNet --model-dir GoogLeNet --force
python3 trainer register --model-name AlexNet --model-dir AlexNet --force
```

2. Validate command resolution without training:

```bash
python3 trainer train GoogLeNet --dry-run
python3 trainer train AlexNet --dry-run
```

3. Run a short smoke train for each model:

```bash
python3 trainer train GoogLeNet --version-name googlenet_smoke --train-samples 2048 --val-samples 256
python3 trainer train AlexNet --version-name alexnet_smoke --train-samples 2048 --val-samples 256
```

4. Add `trainer_params.json` inside model folders if you want project-local defaults that do not require editing Python files.

5. (Optional) Add a third model by creating `<ModelFolder>/trainer_model.py` subclassing `Model`, then register it.
