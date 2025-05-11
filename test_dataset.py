from dataset import ChangeDetectionDataset
from torch.utils.data import DataLoader
from main import custom_collate

root_dir = './data/oscddataset'
cities = ['abudhabi', 'aguasclaras', 'beihai']  # Test with a few cities

try:
    dataset = ChangeDetectionDataset(root_dir, cities, augment=False, require_mask=True)
    print(f"Dataset size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    for i, (img1, img2, mask) in enumerate(loader):
        print(f"Sample {i}: img1 shape={img1.shape}, img2 shape={img2.shape}, mask shape={mask.shape}")
        if i >= 2:  # Limit to 3 samples
            break
except Exception as e:
    print(f"Error: {e}")