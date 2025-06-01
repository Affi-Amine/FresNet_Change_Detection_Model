import os
from PIL import Image
import numpy as np

# Define the root directory for labels
label_root = "./data/oscddataset/train_labels/Onera Satellite Change Detection dataset - Train Labels"

# List of cities with labels (training and validation cities)
cities = [
    "abudhabi", "aguasclaras", "beihai", "beirut", "bercy",
    "bordeaux", "cupertino", "hongkong", "mumbai", "nantes",
    "paris", "pisa", "rennes", "saclay_e"
]

for city in cities:
    label_path = os.path.join(label_root, city, "cm", "cm.png")
    fixed_label_path = os.path.join(label_root, city, "cm", "cm_fixed.png")

    if os.path.exists(label_path):
        # Load the label
        label = Image.open(label_path)
        label_array = np.array(label)

        # Convert to single-channel if necessary
        if label_array.ndim > 2:
            label_array = label_array[:, :, 0]  # Take the first channel
        label_array = (label_array > 0).astype(np.uint8) * 255  # Ensure binary (0 or 255)

        # Save the fixed label
        fixed_label = Image.fromarray(label_array, mode='L')  # 'L' mode for grayscale
        fixed_label.save(fixed_label_path)
        print(f"Fixed label for {city}: {label_array.shape}")
    else:
        print(f"Label not found for {city}: {label_path}")