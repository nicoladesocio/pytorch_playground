"""Dataset loader with caching support for partial downloads."""
import os
import json
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import WeightedRandomSampler
import random


def load_dataset_subset(
    dataset_name,
    cache_root="/workspace/Data/datasets/",
    train_split="train",
    val_split="validation",
    train_samples=1000,
    val_samples=100
):
    """
    Load a subset of any HuggingFace dataset with efficient caching.
    """
    cache_dir = os.path.join(cache_root, dataset_name)
    
    train_cache = os.path.join(cache_dir, f"{train_split}_subset_{train_samples}")
    val_cache = os.path.join(cache_dir, f"{val_split}_subset_{val_samples}")
    metadata_file = os.path.join(cache_dir, f"metadata_{train_samples}_{val_samples}.json")
    
    # Check metadata file first to avoid loading datasets
    cache_valid = False
    train_dataset = None
    validation_dataset = None
    
    if os.path.exists(metadata_file) and os.path.exists(train_cache) and os.path.exists(val_cache):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if (metadata.get('train_samples') == train_samples and 
                metadata.get('val_samples') == val_samples):
                print(f"Loading from cached subset (train: {train_samples}, val: {val_samples})...")
                train_dataset = HFDataset.load_from_disk(train_cache)
                validation_dataset = HFDataset.load_from_disk(val_cache)
                cache_valid = True
            else:
                print(f"Metadata mismatch, re-downloading...")
        except Exception as e:
            print(f"Error reading metadata: {e}")
            print("Re-downloading...")
    
    if not cache_valid:
        print(f"Downloading {dataset_name} subset ({train_samples} train, {val_samples} val samples)...")
        print("This will take a few minutes, only once...")
        
        # Stream the dataset lazily
        dataset = load_dataset(dataset_name, streaming=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 1. Create a generator function that yields items one by one.
        # Shuffle with a buffer before taking N samples: many HF image datasets
        # are shard-ordered by class, so a plain `.take(n)` on the raw stream
        # would silently cache only the first few classes instead of a random
        # cross-section of all classes.
        SHUFFLE_BUFFER_SIZE = 10_000

        def sample_generator(split_name, num_samples):
            shuffled_split = dataset[split_name].shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
            for i, sample in enumerate(shuffled_split.take(num_samples)):
                yield sample
                if (i + 1) % 1024 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples from {split_name}...")

        # 2. Use from_generator to write to disk in chunks, keeping RAM flat
        print("Processing train subset...")
        train_dataset = HFDataset.from_generator(
            sample_generator,
            features=dataset[train_split].features, # Preserves image/audio metadata types
            gen_kwargs={"split_name": train_split, "num_samples": train_samples}
        )
        print("Saving train subset to cache...")
        train_dataset.save_to_disk(train_cache)
        
        print("Processing val subset...")
        validation_dataset = HFDataset.from_generator(
            sample_generator,
            features=dataset[val_split].features,
            gen_kwargs={"split_name": val_split, "num_samples": val_samples}
        )
        print("Saving val subset to cache...")
        validation_dataset.save_to_disk(val_cache)
        
        # Save metadata
        metadata = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'dataset_name': dataset_name
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        print("Cached for future runs!")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    
    return train_dataset, validation_dataset


def create_subset_sampler(dataset, max_num_classes, target_classes = None):

    if target_classes is None:
        all_unique_labels = dataset.unique("label")

        target_classes = set(random.sample(
            all_unique_labels, 
            min(max_num_classes, len(all_unique_labels))
        ))

    # This maps indices to weights: 1 for target classes, 0 otherwise
    weights = [1.0 if x in target_classes else 0.0 for x in dataset["label"]]
    
    # Only sample as many times as there are in-class samples to avoid
    # excessive repetition (e.g. 51200 samples but only ~3300 in-class)
    num_in_class = int(sum(weights))
    if num_in_class == 0:
        raise ValueError(
            f"None of the {len(dataset)} samples in this dataset split belong to "
            f"target_classes={sorted(target_classes)}. Check that the dataset was "
            "cached/loaded correctly."
        )
    
    return WeightedRandomSampler(
        weights=weights,
        # One pass over the in-class samples, no repeats: with replacement=True and
        # num_samples=len(dataset) this used to resample the same ~num_in_class
        # images ~15x per epoch (since ~94% of weights are 0), which inflates the
        # apparent epoch size without adding new information.
        num_samples=num_in_class,
        replacement=False,
    ), target_classes