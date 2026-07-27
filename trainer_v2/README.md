# Trainer CLI (v2)

`trainer_v2` is a CLI-oriented training orchestrator for repeatable deep learning experiments.

It should allow you to:

- Register datasets with deterministic split and transform metadata
- Register model families and their tunable parameter schemas
- Register immutable model versions (configuration snapshots)
- Register training pipelines (runtime and optimization behavior)
- Launch and track training runs by composing dataset + model version + pipeline

# Design Goals

- Reproducibility first: every run must be reconstructable from stored metadata
- Composability: datasets, model versions, and pipelines can be mixed without code edits
- Traceability: artifacts, logs, metrics, and config are linked by run id
- Minimal friction: simple CLI commands, predictable folder structure

# Core Entities

## Dataset

Represents one data source plus the exact preprocessing/split policy.

Required fields:

- `name`
- `description`
- `source` (local path, URL, or loader id)
- `split` (`train/val/test` sizes or ratios)
- `subsampler` (optional)
- `transforms` (optional, ordered)
- `debug_export_path` (optional)
- `seed` (for deterministic split/subsampling)

## Model

Represents a model family and implementation entrypoint.

Required fields:

- `name`
- `description`
- `implementation` (module path/class name or file reference)
- `parameter_schema` (list of tunable parameters with type/default/range)

## Model Version

Represents an immutable instance of a model configuration.

Required fields:

- `model_name`
- `version_name`
- `version_description`
- `model_parameters` (validated against `parameter_schema`)
- `parent_checkpoint` (optional)

Note: after creation, `model_parameters` should be treated as immutable.

## Training Pipeline

Represents runtime strategy and optimization behavior.

Required fields:

- `name`
- `description`
- `trainer_entrypoint` (module or file)
- `batch_size`
- `optimizer` settings
- `scheduler` settings (optional)
- `sampler_policy` (optional)
- `accelerator` (`cpu`, `gpu`, etc.)
- `precision` (for example `32`, `16-mixed`, `bf16-mixed`)
- `max_epochs`
- `seed`

## Training Run

Represents one executable experiment instance.

Required fields:

- `run_name`
- `run_description`
- `dataset_ref`
- `model_ref`
- `model_version_ref`
- `pipeline_ref`
- `status` (`created`, `running`, `failed`, `completed`)
- `artifacts_path`

Expected outputs:

- checkpoints
- tensorboard logs
- serialized run config
- summary metrics (for example best validation accuracy/loss)

# Artifact Layout

Recommended hierarchy:

```text
artifacts/
	models/
		<model_name>/
			runs/
				<run_id>_<run_name>/
					config.yaml
					checkpoints/
					logs/
						tensorboard/
					metrics.json
```

This keeps logs grouped by model while still isolating each run.

# Requirements

## Functional Requirements

1. The CLI must support create/read/list operations for all entities: dataset, model, model version, pipeline, run.
2. The CLI must validate references before creating a training run (dataset/model/model version/pipeline must exist).
3. The CLI must validate model version parameters against the model parameter schema.
4. The system must persist all entity definitions as serializable config (YAML or JSON).
5. A run must persist an immutable execution snapshot (resolved dataset/model version/pipeline config).
6. Training execution must emit checkpoints and scalar metrics to deterministic paths.
7. The run lifecycle must be queryable (`created`, `running`, `failed`, `completed`).
8. TensorBoard logging must be enabled per run and grouped by model.
9. The CLI must support resume from checkpoint for compatible runs.
10. The CLI must provide a dry-run mode that validates config resolution without training.

## Non-Functional Requirements

1. Reproducibility: seed-controlled splits, sampling, and training behavior must be configurable and recorded.
2. Observability: each run must have timestamped logs, final status, and error traces on failure.
3. Extensibility: adding a new model or pipeline should not require changing CLI core logic.
4. Usability: command names and argument contracts should stay stable and discoverable (`--help` everywhere).
5. Reliability: partial failures should not corrupt run metadata; failed runs remain inspectable.
6. Performance: startup and config validation should remain fast for iterative experimentation.

## Suggested CLI Surface (Minimal)

```text
trainer dataset add|list|show
trainer model add|list|show
trainer model-version add|list|show
trainer pipeline add|list|show
trainer run create|start|status|list|resume
```

## Acceptance Criteria

1. A user can define one dataset, one model, one model version, and one pipeline entirely from CLI.
2. A user can launch a run by references only (no code modification).
3. A completed run stores config, checkpoints, and logs in the expected artifact hierarchy.
4. Running the same configuration with the same seed produces comparable metrics behavior.

