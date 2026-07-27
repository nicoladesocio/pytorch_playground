import model
import argparse
import torch
from torch.utils.data import DataLoader
import PIL.Image as Image
import os

import utils.helpers as helpers

import custum_datasets.images_folder_dataset as images_folder_dataset

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="Test AlexNet model")
    argument_parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for AlexNet")
    argument_parser.add_argument("--input_images_folder", type=str, default="/workspace/Data/images", help="Path to input images folder")
    argument_parser.add_argument("--width", type=int, default=227, help="Size to resize images (AlexNet uses 227x227)")
    argument_parser.add_argument("--height", type=int, default=227, help="Size to resize images (AlexNet uses 227x227)")
    argument_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = argument_parser.parse_args()

    print("Testing AlexNet model")
    model = model.AlexNet(num_classes=args.num_classes)
    print("Model created successfully!")

    # Create dataset and dataloader
    dataset = images_folder_dataset.ImagesFolderDataset(args.input_images_folder, image_size=(args.height, args.width))
    print(f"Found {len(dataset)} valid images")
    
    if len(dataset) > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Process batches
        for batch_idx, (batch_tensor, filenames) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}: shape {batch_tensor.shape}, files: {filenames}")
            
            # Run inference on the batch
            output = model(batch_tensor)
            print(f"Output shape: {output.shape}")
    else:
        print("No valid images found")