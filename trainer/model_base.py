import importlib.util
import inspect
import json
from pathlib import Path


class Model:
    name = "base"
    train_script = "train.py"
    default_parameters = {
        "version_name": "default",
    }

    def __init__(self, model_dir: Path, project_root: Path):
        self.model_dir = Path(model_dir).resolve()
        self.project_root = Path(project_root).resolve()

    def _load_folder_parameters(self) -> dict:
        config_path = self.model_dir / "trainer_params.json"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON object in {config_path}")
        return data

    def get_default_parameters(self) -> dict:
        folder_defaults = self._load_folder_parameters()
        merged = dict(self.default_parameters)
        merged.update(folder_defaults)
        return merged

    def resolve_parameters(self, explicit_overrides: dict, set_overrides: dict) -> dict:
        resolved = self.get_default_parameters()
        resolved.update(explicit_overrides)
        resolved.update(set_overrides)
        return resolved

    @staticmethod
    def _to_cli_args(params: dict) -> list[str]:
        args = []
        for key, value in params.items():
            if value is None:
                continue
            arg_name = f"--{key}"
            if isinstance(value, bool):
                if value:
                    args.append(arg_name)
                continue
            if isinstance(value, list):
                for item in value:
                    args.extend([arg_name, str(item)])
                continue
            args.extend([arg_name, str(value)])
        return args

    def build_train_command(
        self,
        python_executable: str,
        resolved_parameters: dict,
        passthrough_args: list[str],
    ) -> list[str]:
        train_script_path = self.model_dir / self.train_script
        if not train_script_path.exists():
            raise FileNotFoundError(f"Train script not found: {train_script_path}")

        return [
            python_executable,
            str(train_script_path),
            *self._to_cli_args(resolved_parameters),
            *passthrough_args,
        ]


def _load_module(module_file: Path):
    module_name = f"trainer_dynamic_{module_file.parent.name}_{module_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_class(module_file: Path, class_name: str | None = None):
    module = _load_module(module_file)

    if class_name:
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ValueError(f"Class '{class_name}' not found in {module_file}")
        if not inspect.isclass(cls) or not issubclass(cls, Model) or cls is Model:
            raise ValueError(f"Class '{class_name}' is not a valid Model subclass")
        return cls

    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Model) and obj is not Model:
            candidates.append(obj)

    if not candidates:
        raise ValueError(f"No Model subclass found in {module_file}")
    if len(candidates) > 1:
        names = ", ".join(sorted(cls.__name__ for cls in candidates))
        raise ValueError(
            f"Multiple Model subclasses found in {module_file}: {names}. "
            "Use --model-class to specify one."
        )
    return candidates[0]
