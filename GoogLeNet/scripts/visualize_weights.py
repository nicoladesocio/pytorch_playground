import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def resolve_weight_key(state_dict, requested_key=None):
    if requested_key:
        if requested_key in state_dict:
            return requested_key
        raise KeyError(f"Requested weight key not found: {requested_key}")

    # Preferred defaults for this GoogLeNet implementation.
    candidate_keys = [
        "model.conv1.weight",
        "conv1.weight",
    ]
    for key in candidate_keys:
        if key in state_dict:
            return key

    # Fallback: first 4D tensor in the state dict (likely conv weights).
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 4:
            return key

    raise ValueError("No 4D convolution weight tensor found in checkpoint state_dict")


def normalize_per_filter(weights):
    # Normalize each output channel independently to improve visibility.
    # weights shape: [out_channels, in_channels, h, w]
    w = weights.clone()
    flat = w.view(w.shape[0], -1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    denom = torch.clamp(maxs - mins, min=1e-8)
    return (w - mins) / denom


def choose_grid(n_items):
    n_cols = int(math.ceil(math.sqrt(n_items)))
    n_rows = int(math.ceil(n_items / n_cols))
    return n_rows, n_cols


def main():
    parser = argparse.ArgumentParser(description="Visualize GoogLeNet convolution filters")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--weight_key",
        type=str,
        default=None,
        help="Weight tensor key in checkpoint state_dict (default auto-detect)",
    )
    parser.add_argument(
        "--max_filters",
        type=int,
        default=64,
        help="Maximum number of filters to plot",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/Data/images/googlenet_weights.png",
        help="Output image path",
    )
    parser.add_argument(
        "--list_keys",
        action="store_true",
        help="List all state_dict keys and exit",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    if args.list_keys:
        print("Available state_dict keys:")
        for key in state_dict.keys():
            print(f"  {key}")
        return

    weight_key = resolve_weight_key(state_dict, args.weight_key)
    weights = state_dict[weight_key].detach().cpu()

    if weights.ndim != 4:
        raise ValueError(
            f"Selected key '{weight_key}' is not a convolution tensor. "
            f"Expected 4D, got shape {tuple(weights.shape)}"
        )

    out_channels, in_channels, _, _ = weights.shape
    n_filters = min(args.max_filters, out_channels)
    weights = weights[:n_filters]
    weights_normalized = normalize_per_filter(weights)

    n_rows, n_cols = choose_grid(n_filters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if not isinstance(axes, (list, tuple)):
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = list(axes)

    for i, ax in enumerate(axes):
        if i < n_filters:
            if in_channels >= 3:
                img = weights_normalized[i][:3].permute(1, 2, 0)
                ax.imshow(img)
            else:
                ax.imshow(weights_normalized[i][0], cmap="gray")
            ax.set_title(str(i), fontsize=7)
        ax.axis("off")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.suptitle(f"{weight_key} shape={tuple(state_dict[weight_key].shape)}", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches="tight")
    print(f"Saved weights visualization to: {args.output_path}")
    print(f"Used key: {weight_key}")


if __name__ == "__main__":
    main()
