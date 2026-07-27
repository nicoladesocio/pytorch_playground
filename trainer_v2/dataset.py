from utils.datasets_utils import load_dataset_subset

DATASET_CACHE_ROOT = "/workspace/Data/datasets/"

DATASET_REGISTRY = {
    "imagenet-1k-51200-2048": {
        "dataset-name": "imagenet-1k",
        "description": "Subset of ImageNet with 1000 classes, used for training and validation. The full dataset is large, so we use a subset of it for faster experimentation.",
        "train_samples": 51200,
        "val_samples": 2048,
    }
}

def dataset(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry.")

    train_dataset, validation_dataset = load_dataset_subset(
            dataset_name=DATASET_REGISTRY[dataset_name]["dataset-name"],
            cache_root=DATASET_CACHE_ROOT,
            train_samples=DATASET_REGISTRY[dataset_name]["train_samples"],
            val_samples=DATASET_REGISTRY[dataset_name]["val_samples"]
        )
    return train_dataset, validation_dataset