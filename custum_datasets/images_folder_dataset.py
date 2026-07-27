"""Image dataset for loading and preprocessing images."""
import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms


def is_valid_image(path):
    """Check if a file is a valid image."""
    try:
        Image.open(path).verify()
        return True
    except (IOError, SyntaxError):
        return False


class ImagesFolderDataset(Dataset):
    """Dataset for loading images from a folder with optional resizing."""
    
    def __init__(self, image_folder, image_size=None):
        """
        Args:
            image_folder (str): Path to folder containing images
            image_size (tuple or int, optional): Size to resize images to (H, W) or single int for square.
                                                  If None, no resizing is performed.
        """
        self.image_folder = image_folder
        
        # Get all valid image files
        self.image_files = [
            f for f in os.listdir(image_folder) 
            if is_valid_image(os.path.join(image_folder, f))
        ]
        
        # Setup transforms
        transform_list = []
        if image_size:
            transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.ToTensor())

        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        image = self.transform(image)
        
        return image, self.image_files[idx]  # Return tensor and filename
