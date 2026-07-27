import torch
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

# Use a non-display backend that works in Docker, but support showing in Jupyter/IPython
matplotlib.use('Agg')

argument_parser = argparse.ArgumentParser(description="Visualize AlexNet first layer weights")
argument_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
args = argument_parser.parse_args()

# 1. Load the checkpoint
checkpoint = torch.load(args.checkpoint_path)
state_dict = checkpoint['state_dict']

# 2. Extract the first layer weights (net.0.weight)
# Note: Lightning prefixes names with 'model.', so check your keys
first_layer_weights = state_dict['model.net.0.weight'].cpu()

# 3. Normalize weights to [0, 1] for visualization
w_min, w_max = first_layer_weights.min(), first_layer_weights.max()
weights_normalized = (first_layer_weights - w_min) / (w_max - w_min)

# 4. Plot the 96 filters
fig, axes = plt.subplots(8, 12, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    if i < 96:
        # Weights are (out_channels, in_channels, h, w) -> (96, 3, 11, 11)
        # We transpose to (11, 11, 3) for imshow
        img = weights_normalized[i].permute(1, 2, 0)
        ax.imshow(img)
    ax.axis('off')

# Save the figure instead of showing it
output_path = "/workspace/Data/images/alexnet_first_layer_weights.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to {output_path}")

# Try to display in Jupyter/IPython environment, otherwise suggest opening the file
try:
    from IPython.display import Image, display
    display(Image(filename=output_path))
except ImportError:
    # Fallback: try to open the file with xdg-open or similar
    try:
        os.system(f"xdg-open {output_path}")
    except:
        print(f"To view the image, open: {output_path}")

# Also call plt.show() for environments where it works
plt.show()