import importlib

def load_module(module_name, class_name, **init_kwargs):
    try:
        module = importlib.import_module(module_name)
        module_class = getattr(module, class_name)
        return module_class(**init_kwargs)
    except ImportError as exc:
        raise ImportError(f"Could not import module '{module_name}'.") from exc
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' does not define expected class '{class_name}'.") from exc