import os

DATASET_CACHE_ROOT = "/workspace/Data/checkpoints/"

# Sort by (epoch_number, version_number) to get latest epoch first, then latest version
def __get_epoch_and_version(filepath):
    try:
        # Extract epoch and version from "alexnet_epoch_epoch=XX(-vy).ckpt"
        # Find the part after "epoch=" and remove the .ckpt extension
        start_idx = filepath.find("epoch=") + len("epoch=")
        remainder = filepath[start_idx:].replace(".ckpt", "")
        
        # Split by "-v" to separate epoch from version
        parts = remainder.split("-v")
        epoch_num = int(parts[0])
        
        # If there's a version suffix, extract it; otherwise treat as version 0
        if len(parts) > 1:
            version_num = int(parts[1])
        else:
            version_num = 0
        
        return (epoch_num, version_num)
    except (IndexError, ValueError):
        return (-1, -1)

def load_latest_checkpoint(model_name, run_name):
    """Load the latest checkpoint for a model and run.
    
    Prioritizes by version (latest first), then by epoch (latest first).
    
    Args:
        model_name: Name of the model (e.g., 'AlexNet')
        run_name: Name/prefix of the run (e.g., 'baseline')
    
    Returns:
        Full path to the latest checkpoint, or None if not found.
    """
    checkpoint_dir = os.path.join(DATASET_CACHE_ROOT, model_name)
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(run_name)]
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}' for run '{run_name}'.")
        return None
    
    # Pair each checkpoint file with its (epoch, version) tuple
    checkpoint_with_info = [(f, __get_epoch_and_version(f)) for f in checkpoint_files]
    
    # Filter out files with invalid epoch/version (marked as (-1, -1))
    valid_checkpoints = [(f, (e, v)) for f, (e, v) in checkpoint_with_info if (e, v) != (-1, -1)]
    
    if not valid_checkpoints:
        print(f"No valid checkpoint files found in '{checkpoint_dir}' for run '{run_name}'.")
        return None
    
    # Sort by version (descending), then by epoch (descending)
    # This prioritizes latest version first, then latest epoch within that version
    latest_file, _ = max(valid_checkpoints, key=lambda x: (x[1][1], x[1][0]))
    
    return os.path.join(checkpoint_dir, latest_file)