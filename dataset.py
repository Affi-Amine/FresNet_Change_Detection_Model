# dataset.py

import os
import warnings
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Fixed size for all images/masks
FIXED_SIZE = (256, 256)

# Normalization values (ImageNet / common RGB)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Base transforms for both training and evaluation
base_transform = transforms.Compose([
    transforms.Resize(FIXED_SIZE, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    normalize,
])

# Additional augmentations for training
spatial_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(90),
])

color_transforms = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)

# Transform for mask only (resize + tensor, no normalization)
mask_transform = transforms.Compose([
    transforms.Resize(FIXED_SIZE, interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, cities, augment=True, require_mask=True):
        """
        root_dir/
          images/OSCD/<city>/pair/img1.png, img2.png
          train_labels/OSCD/<city>/cm/<city>-cm.tif or .png
        """
        self.augment = augment
        self.require_mask = require_mask
        self.samples = []
        self.pos_weight = 0  # To track class balance
        self.total_pixels = 0

        for city in cities:
            pair_dir = os.path.join(root_dir, 'images', 'OSCD', city, 'pair')
            mask_dir = os.path.join(root_dir, 'train_labels', 'OSCD', city, 'cm')
            tif = os.path.join(mask_dir, f"{city}-cm.tif")
            png = os.path.join(mask_dir, "cm.png")
            mask_file = png if os.path.isfile(png) else tif

            if not os.path.isdir(pair_dir):
                warnings.warn(f"[dataset] Missing pair/ for {city}: {pair_dir}")
                continue

            img1 = os.path.join(pair_dir, 'img1.png')
            img2 = os.path.join(pair_dir, 'img2.png')
            if not os.path.isfile(img1) or not os.path.isfile(img2):
                warnings.warn(f"[dataset] Missing img1/img2 in {pair_dir}")
                continue

            if self.require_mask:
                if not os.path.isfile(mask_file):
                    warnings.warn(f"[dataset] Missing mask for {city}: {mask_file}")
                    continue
                # Calculate class balance
                mask = np.array(Image.open(mask_file).convert('L'))
                self.total_pixels += mask.size
                self.pos_weight += np.sum(mask > 0)

            self.samples.append((img1, img2, mask_file if os.path.isfile(mask_file) else None))

        if not self.samples:
            raise RuntimeError("No valid samples found – check your `pair` and `cm` paths.")

        # Calculate positive class weight for loss function
        if self.require_mask and self.total_pixels > 0:
            neg_samples = self.total_pixels - self.pos_weight
            self.pos_weight = neg_samples / self.pos_weight if self.pos_weight > 0 else 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, mask_path = self.samples[idx]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Resize images first to ensure consistent size
        img1 = transforms.Resize(FIXED_SIZE, interpolation=Image.BILINEAR)(img1)
        img2 = transforms.Resize(FIXED_SIZE, interpolation=Image.BILINEAR)(img2)

        # Convert to tensors
        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)

        # Apply augmentations during training
        if self.augment:
            # Stack images for spatial transforms to keep them aligned
            stacked = torch.cat([img1, img2], dim=0)
            # Apply same spatial transforms to both images
            stacked = spatial_transforms(stacked)
            img1, img2 = torch.split(stacked, 3, dim=0)
            
            # Apply color transforms separately
            img1 = color_transforms(img1)
            img2 = color_transforms(img2)

        # Apply normalization
        img1 = normalize(img1)
        img2 = normalize(img2)

        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize(FIXED_SIZE, interpolation=Image.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)
            if self.augment:
                mask = spatial_transforms(mask)
            # Convert to binary mask
            mask = (mask > 0.5).long().squeeze(0)
        else:
            mask = torch.zeros(FIXED_SIZE, dtype=torch.long)

        # Ensure all tensors are contiguous
        return img1.contiguous(), img2.contiguous(), mask.contiguous()

    def get_pos_weight(self):
        """Returns the positive class weight for weighted loss"""
        return self.pos_weight
