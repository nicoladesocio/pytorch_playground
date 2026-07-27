"""Shared HuggingFace-dataset wrapper used by the AlexNet and GoogLeNet training
pipelines (previously duplicated as AlexNet/alexnet_dataset.py and
GoogLeNet/googlenet_dataset.py - merged here since both were identical aside
from GoogLeNet's optional debug-image dump)."""
import hashlib
import json
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset


class ImageNetSubsetDataset(TorchDataset):
    """Wrap a HuggingFace image-classification dataset with 2012-style (AlexNet
    paper) data augmentation, optional class-id remapping to a contiguous
    0..N-1 range, and optional debug image dumping.
    """

    def __init__(
        self,
        hf_dataset,
        image_size=224,
        is_training=True,
        target_classes=None,
        debug_images_dir=None,
    ):
        self.hf_dataset = hf_dataset
        self.debug_images_dir = debug_images_dir
        # Build mapping from original class id -> contiguous index (0..N-1)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)} if target_classes is not None else None

        if self.debug_images_dir is not None:
            self.hashes_file = os.path.join(debug_images_dir, ".saved_image_hashes.json")
            self.saved_image_hashes = self._load_hashes()

        # Note: Original paper resized to 256, then took a 224 crop
        if is_training:
            transform_list = [
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ]
        else:
            # For validation, we typically just take the Center Crop
            transform_list = [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]

        # Standard ImageNet normalization (applied to [0,1] images)
        transform_list.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        self.transform = transforms.Compose(transform_list)

    def _load_hashes(self):
        """Load previously saved image hashes from file."""
        if os.path.exists(self.hashes_file):
            try:
                with open(self.hashes_file, 'r') as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError):
                return set()
        return set()

    def _save_hashes(self):
        """Persist saved image hashes to file."""
        os.makedirs(self.debug_images_dir, exist_ok=True)
        with open(self.hashes_file, 'w') as f:
            json.dump(list(self.saved_image_hashes), f)

    def _compute_image_hash(self, image_tensor):
        """Compute SHA256 hash of image tensor."""
        image_bytes = image_tensor.cpu().numpy().tobytes()
        return hashlib.sha256(image_bytes).hexdigest()

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # Ensure image is in RGB
        image = item["image"].convert("RGB")
        label = item.get("label", item.get("labels"))

        # Remap label to contiguous index if a mapping is provided
        if self.class_to_idx is not None:
            label = self.class_to_idx[label]

        # Apply the 2012-style transforms (including normalization)
        image_tensor = self.transform(image)

        # Optionally save the (unnormalized) image for debugging, deduped by hash
        if self.debug_images_dir is not None:
            image_hash = self._compute_image_hash(image_tensor)
            if image_hash not in self.saved_image_hashes:
                try:
                    from torchvision.utils import save_image
                    label_dir = os.path.join(self.debug_images_dir, str(label))
                    os.makedirs(label_dir, exist_ok=True)

                    # Unnormalize for visualization: convert back to [0,1] range
                    inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_vis = image_tensor * inv_std + inv_mean
                    img_vis = torch.clamp(img_vis, 0, 1)

                    save_path = os.path.join(label_dir, f"img_{image_hash[:8]}.jpg")
                    save_image(img_vis, save_path)

                    self.saved_image_hashes.add(image_hash)
                    self._save_hashes()
                except Exception as e:
                    print(f"Warning: Failed to save image {idx}: {e}")

        return image_tensor, label
