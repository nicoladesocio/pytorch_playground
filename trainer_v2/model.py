
from load_module import load_module

MODEL_REGISTRY = {
    "AlexNet": {
        "description": "AlexNet model architecture",
        "module": "AlexNet.model"
    },
    "GoogLeNet": {
        "description": "GoogLeNet model architecture",
        "module": "GoogLeNet.model"
    }
}

VERSION_REGISTRY = {
    "AlexNet": { 
        "v1": {
            "description": "Baseline AlexNet with default hyperparameters.",
            "parameters": {
                "num_classes": 64,
                "target_classes": None
            }
        },
    },
    "GoogLeNet": {
        "v1": {
            "description": "Baseline GoogLeNet with default hyperparameters.",
            "parameters": {
                "num_classes": 1000,
                "target_classes": None
            }
        },
    }
}

def load_model(model_name, version_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    
    if model_name not in VERSION_REGISTRY or version_name not in VERSION_REGISTRY[model_name]:
        raise ValueError(f"Version '{version_name}' not found for model '{model_name}'.")
    
    module_path = MODEL_REGISTRY[model_name]["module"]
    parameters = VERSION_REGISTRY[model_name][version_name]["parameters"]
    class_name = f"{model_name}Lightning"

    try:
        model_instance = load_module(module_path, class_name, **parameters)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"Could not load model class '{class_name}' from module '{module_path}'."
        ) from exc

    return model_instance


def model(model_name, version_name):
    return load_model(model_name, version_name)