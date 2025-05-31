from PIL import Image
import numpy as np

label = Image.open("./data/oscddataset/train_labels/Onera Satellite Change Detection dataset - Train Labels/abudhabi/cm/cm.png")
print(np.array(label).shape)