import argparse
import json
import subprocess
import sys
from pathlib import Path

from trainer.model_base import Model, load_model_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"




def _read_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(registry: dict) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)


def _parse_set_values(items: list[str]) -> dict:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Use KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        raw_value = raw_value.strip()
        try:
            parsed[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed[key] = raw_value
    return parsed


def _resolve_model_dir(model_name: str, model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir).resolve()
    return (PROJECT_ROOT / model_name).resolve()


def handle_register(args: argparse.Namespace) -> int:
    model_dir = _resolve_model_dir(args.model_name, args.model_dir)
    module_file = model_dir / "trainer_model.py"
    if not module_file.exists():
        raise FileNotFoundError(
            f"Model registration file not found: {module_file}. "
            "Add trainer_model.py in the model folder."
        )

    model_class = load_model_class(module_file, class_name=args.model_class)
    model_obj: Model = model_class(model_dir=model_dir, project_root=PROJECT_ROOT)

    registry = _read_registry()
    if args.model_name in registry and not args.force:
        raise ValueError(
            f"Model '{args.model_name}' already registered. "
            "Use --force to overwrite."
        )

    registry[args.model_name] = {
        "name": model_obj.name,
        "model_dir": str(model_dir),
        "module_file": str(module_file),
        "class_name": model_class.__name__,
        "default_parameters": model_obj.get_default_parameters(),
        "train_script": model_obj.train_script,
    }
    _write_registry(registry)

    print(f"Registered model '{args.model_name}' from {module_file}")
    print("Default train parameters:")
    for key, value in model_obj.get_default_parameters().items():
        print(f"  - {key}: {value}")
    return 0


def handle_train(args: argparse.Namespace, passthrough: list[str]) -> int:
    registry = _read_registry()
    if args.model_name not in registry:
        raise KeyError(
            f"Model '{args.model_name}' is not registered. "
            "Run: python trainer register --model-name <name>"
        )

    record = registry[args.model_name]
    module_file = Path(record["module_file"]).resolve()
    model_class = load_model_class(module_file, class_name=record["class_name"])
    model_obj: Model = model_class(
        model_dir=Path(record["model_dir"]).resolve(),
        project_root=PROJECT_ROOT,
    )

    explicit_overrides = {
        "version_name": args.version_name,
        "num_classes": args.num_classes,
        "checkpoint_path": args.checkpoint_path,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "batch_size": args.batch_size,
    }
    explicit_overrides = {k: v for k, v in explicit_overrides.items() if v is not None}
    set_overrides = _parse_set_values(args.set)

    resolved = model_obj.resolve_parameters(
        explicit_overrides=explicit_overrides,
        set_overrides=set_overrides,
    )
    command = model_obj.build_train_command(
        python_executable=sys.executable,
        resolved_parameters=resolved,
        passthrough_args=passthrough,
    )

    print("Executing training command:")
    print(" ".join(command))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model registry and trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="Register a model folder")
    register_parser.add_argument("--model-name", required=True, help="Registry key for this model")
    register_parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to the model folder. Defaults to sibling folder PROJECT_ROOT/model-name",
    )
    register_parser.add_argument(
        "--model-class",
        default=None,
        help="Model class name inside trainer_model.py. If omitted, auto-detect subclass.",
    )
    register_parser.add_argument("--force", action="store_true", help="Overwrite existing registration")

    train_parser = subparsers.add_parser("train", help="Train a registered model")
    train_parser.add_argument("model_name", help="Registered model name")
    train_parser.add_argument("--version-name", default=None, help="Experiment/version name")
    train_parser.add_argument("--num-classes", type=int, default=None)
    train_parser.add_argument("--checkpoint-path", default=None)
    train_parser.add_argument("--train-samples", type=int, default=None)
    train_parser.add_argument("--val-samples", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any parameter with KEY=VALUE. VALUE accepts JSON literals.",
    )
    train_parser.add_argument("--dry-run", action="store_true", help="Only print command")

    return parser


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if args.command == "register":
        return handle_register(args)
    if args.command == "train":
        return handle_train(args, passthrough)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
