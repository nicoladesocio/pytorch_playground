"""Helper utilities for model training and evaluation."""

def print_info(message):
    """Print formatted information message."""
    print(f"[INFO] {message}")

def calculate_accuracy(predictions, targets):
    """Calculate accuracy from predictions and targets."""
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = 100 * correct / total
    return accuracy
