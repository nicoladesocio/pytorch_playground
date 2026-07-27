"""Model loading utilities."""
import os
import glob
import torch


def get_latest_checkpoint(checkpoint_dir, model_prefix="alexnet_epoch"):
    """
    Find the latest checkpoint in the directory by epoch number, then by version.
    Handles checkpoint filenames with version suffixes like:
    - alexnet_epoch_epoch=02.ckpt (treated as version 0)
    - alexnet_epoch_epoch=02-v1.ckpt
    - alexnet_epoch_epoch=02-v10.ckpt
    
    Returns the checkpoint with the highest epoch, and if multiple exist for the same epoch,
    returns the one with the highest version number.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist.
    """
    # Look for epoch-based checkpoints (pattern: alexnet_epoch_epoch=XX*.ckpt)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, f"{model_prefix}_epoch=*.ckpt"))
    
    if not checkpoints:
        return None
    
    # Sort by (epoch_number, version_number) to get latest epoch first, then latest version
    def get_epoch_and_version(filepath):
        filename = os.path.basename(filepath)
        try:
            # Extract epoch and version from "alexnet_epoch_epoch=XX(-vy).ckpt"
            # Find the part after "epoch=" and remove the .ckpt extension
            start_idx = filename.find("epoch=") + len("epoch=")
            remainder = filename[start_idx:].replace(".ckpt", "")
            
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
    
    latest = max(checkpoints, key=get_epoch_and_version)
    return latest


def load_model_checkpoint(model, checkpoint_root):
    """
    Load model weights from a checkpoint.
    
    Args:
        model: The Lightning model instance to load weights into
        checkpoint_root: Path to the checkpoint directory
        
    Returns:
        model: Model with loaded weights
    """
    if checkpoint_root:
        checkpoint_path = os.path.join(checkpoint_root, "alexnet_final.ckpt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded model parameters from {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, using randomly initialized model")
    else:
        print("No checkpoint root provided, using randomly initialized model")
    
    return model


def save_model_checkpoint(model, checkpoint_root):
    """
    Save model weights to a checkpoint.
    
    Args:
        model: The model instance to save
        checkpoint_root: Directory to save the checkpoint file
    """
    if not checkpoint_root:
        print("No checkpoint root provided, skipping save")
        return

    checkpoint_path = os.path.join(checkpoint_root, "alexnet_final.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
