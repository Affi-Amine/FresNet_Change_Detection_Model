# dataset.py

import os
import warnings
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

FIXED_SIZE = (256, 256)

base_transform = transforms.Compose([
    transforms.Resize(FIXED_SIZE, interpolation=Image.BILINEAR),
])
augmentation = transforms.Compose([
    transforms.Resize(FIXED_SIZE, interpolation=Image.BILINEAR),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
])

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, cities, augment=True, require_mask=True):
        """
        root_dir/
          images/OSCD/<city>/pair/img1.png, img2.png
          train_labels/OSCD/<city>/cm/city-cm.tif or .png
        require_mask: if False (inference), allow missing masks
        """
        self.augment = augment
        self.require_mask = require_mask
        self.samples = []

        for city in cities:
            pair_dir = os.path.join(root_dir, 'images', 'OSCD', city, 'pair')
            mask_dir = os.path.join(root_dir, 'train_labels', 'OSCD', city, 'cm')
            # mask could be either .tif or .png
            tif = os.path.join(mask_dir, f"{city}-cm.tif")
            png = os.path.join(mask_dir, f"{city}-cm.png")
            mask_file = png if os.path.isfile(png) else tif

            if not os.path.isdir(pair_dir):
                warnings.warn(f"[dataset] Missing pair/ for city {city}: {pair_dir}")
                continue

            img1 = os.path.join(pair_dir, 'img1.png')
            img2 = os.path.join(pair_dir, 'img2.png')
            if not os.path.isfile(img1) or not os.path.isfile(img2):
                warnings.warn(f"[dataset] Missing img1/img2 in {pair_dir}")
                continue

            if self.require_mask and not os.path.isfile(mask_file):
                warnings.warn(f"[dataset] Missing mask for {city}: {mask_file}")
                continue

            # if mask not required, allow None
            self.samples.append((img1, img2, mask_file if os.path.isfile(mask_file) else None))

        if not self.samples:
            raise RuntimeError("No valid samples found – check your paths.")

        self.dup_count = len(self.samples)

    def __len__(self):
        return len(self.samples) + (self.dup_count if self.require_mask else 0)

    def __getitem__(self, idx):
        # handle duplicates only when mask required (training)
        if self.require_mask and idx >= len(self.samples):
            idx0 = idx - len(self.samples)
            img1_path, _, _ = self.samples[idx0]
            img1 = Image.open(img1_path)
            img2 = img1.copy()
            mask = Image.fromarray(np.zeros((img1.height, img1.width), dtype=np.uint8))
        else:
            img1_path, img2_path, mask_path = self.samples[idx]
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            if mask_path:
                mask = Image.open(mask_path).convert('L')
            else:
                mask = None

        # transforms
        if self.augment:
            img1 = augmentation(img1)
            img2 = augmentation(img2)
            if mask is not None:
                mask = augmentation(mask)
        else:
            img1 = base_transform(img1)
            img2 = base_transform(img2)
            if mask is not None:
                mask = base_transform(mask)

        # to tensor
        to_t = transforms.functional.to_tensor
        img1 = to_t(img1).float()
        img2 = to_t(img2).float()
        if mask is not None:
            mask = (to_t(mask) > 0.5).long().squeeze(0)
        else:
            mask = torch.zeros(FIXED_SIZE, dtype=torch.long)  # dummy for inference

        return img1, img2, mask
