import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from datetime import datetime

class ChangeDetectionDataset(Dataset):
    def __init__(self, root, cities, transform=None, augment=False, require_mask=False, return_time=False, use_all_bands=False):
        self.root = root
        self.cities = cities
        self.transform = transform
        self.use_all_bands = use_all_bands
        # If no transform is provided, use a default one to convert to tensor
        self.default_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),  # Force square resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common normalization for RGB images
        ]) if transform is None else transform
        self.augment = augment
        self.require_mask = require_mask
        self.return_time = return_time
        self.imgs1 = []
        self.imgs2 = []
        self.labels = []
        self.time_diffs = []
        self.names = []

        # Define augmentation transforms if augment is True
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
        ]) if augment else None

        for city in cities:
            # Image paths
            img1_path = os.path.join(root, "images", "OSCD", city, "pair", "img1.png")
            img2_path = os.path.join(root, "images", "OSCD", city, "pair", "img2.png")
            
            # Label path (use the fixed single-channel label)
            label_path = os.path.join(root, "train_labels", "Onera Satellite Change Detection dataset - Train Labels", city, "cm", "cm_fixed.png")
            
            # Time difference from dates.txt
            dates_path = os.path.join(root, "images", "OSCD", city, "dates.txt")
            if os.path.exists(dates_path):
                with open(dates_path, 'r') as f:
                    dates = f.read().strip().split('\n')
                    # Parse dates in the format 'date_1: YYYYMMDD' and 'date_2: YYYYMMDD'
                    try:
                        date1_str = dates[0].split(": ")[1].strip()  # Extract YYYYMMDD
                        date2_str = dates[1].split(": ")[1].strip()  # Extract YYYYMMDD
                        # Convert to datetime objects
                        date1 = datetime.strptime(date1_str, '%Y%m%d')
                        date2 = datetime.strptime(date2_str, '%Y%m%d')
                        # Compute difference in days and normalize to years
                        time_diff = (date2 - date1).days / 365.25
                    except (IndexError, ValueError, KeyError) as e:
                        print(f"Warning: Could not parse dates for {city}, setting time_diff to 0.0. Error: {e}")
                        time_diff = 0.0
            else:
                time_diff = 0.0

            if os.path.exists(img1_path) and os.path.exists(img2_path):
                if not require_mask or (require_mask and os.path.exists(label_path)):
                    self.imgs1.append(img1_path)
                    self.imgs2.append(img2_path)
                    self.time_diffs.append(time_diff)
                    self.names.append(city)
                    if os.path.exists(label_path):
                        self.labels.append(label_path)
                    else:
                        self.labels.append(None)

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, idx):
        img1_path = self.imgs1[idx]
        img2_path = self.imgs2[idx]
        label_path = self.labels[idx]
        time_diff = self.time_diffs[idx]

        # Load images (default to RGB for now)
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Placeholder for multispectral data loading
        if self.use_all_bands:
            print(f"Warning: use_all_bands=True, but multispectral loading is not implemented. Using RGB instead.")
            # TODO: Implement loading of multispectral bands (e.g., from TIF files)

        # Apply augmentation if enabled
        if self.augment and self.augmentation:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img1 = self.augmentation(img1)
            torch.manual_seed(seed)  # Ensure same augmentation for img2
            img2 = self.augmentation(img2)

        # Apply transform (default or custom)
        img1 = self.default_transform(img1)
        img2 = self.default_transform(img2)

        if label_path:
            label = Image.open(label_path)
            label = np.array(label, dtype=np.uint8)
            # Ensure label is single-channel
            if label.ndim > 2:
                label = label[:, :, 0]  # Take the first channel as a fallback
            # Resize label to match image size
            label = Image.fromarray(label)
            resize_transform = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
            label = resize_transform(label)
            label = np.array(label, dtype=np.uint8)
            # Convert to binary (0 or 1)
            label = (label > 0).astype(np.uint8)
            label = torch.from_numpy(label).long().unsqueeze(0)  # Add channel dimension [1, 256, 256]
            print(f"Label shape after processing: {label.shape}")  # Debug print
        else:
            label = torch.zeros((1, 256, 256), dtype=torch.long, device=img1.device)  # Dummy label with channel dimension

        if self.return_time:
            return img1, img2, label, torch.tensor(time_diff, dtype=torch.float32), self.names[idx]
        return img1, img2, label, self.names[idx]

    def get_pos_weight(self):
        """
        Compute the positive weight for the binary cross-entropy loss based on the class imbalance.
        Returns the ratio of negative to positive pixels in the labels.
        """
        pos_pixels = 0
        neg_pixels = 0

        for label_path in self.labels:
            if label_path is None:
                continue
            label = Image.open(label_path)
            label = np.array(label, dtype=np.uint8)
            # Ensure label is single-channel
            if label.ndim > 2:
                label = label[:, :, 0]  # Take the first channel as a fallback
            # Resize label to match image size for consistency
            label = Image.fromarray(label)
            label = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(label)
            label = np.array(label, dtype=np.uint8)
            # Convert to binary: 0 (no change), 1 (change)
            label = (label > 0).astype(np.uint8)
            pos_pixels += np.sum(label == 1)
            neg_pixels += np.sum(label == 0)

        if pos_pixels == 0:
            return 1.0  # Avoid division by zero; use a neutral weight
        return neg_pixels / pos_pixels