import os
import csv
from typing import List, Tuple

from PIL import Image
import torch
import torchvision
from torch.utils import data

# abspath return the absolute path
# dirname return the directory of the file (remove the file name)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# join the path together
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "banana-detection", "banana-detection")


def read_data_bananas(is_train: bool = True):
    data_dir = DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR not found: {data_dir}. Please place the banana-detection dataset there."
        )
    # for this data set we have:
    # filename label x1 y1 x2 y2
    # labels are always 0 because we only detect bananas
    subset = "banana_train" if is_train else "banana_val"
    # find the label.csv
    csv_path = os.path.join(data_dir, subset, "label.csv")
    
    image_paths, labels = [], []
    
    # with .. as f , f is file and will close automatically
    with open(csv_path, "r", newline="") as f:
        # read by lines
        reader = csv.reader(f)
        # skip the header
        _ = next(reader)
        
        for row in reader:
            image_paths.append(os.path.join(data_dir, subset, "image", row[0]))
            labels.append([int(row[1])] + [float(x) for x in row[2:]])
            
    return image_paths, labels

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train: bool = True, edge_size: int = 256):
        super().__init__()
        self.features, self.labels = read_data_bananas(is_train)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((edge_size, edge_size)),
                torchvision.transforms.ToTensor(),
            ]
        )
        
    def __len__(self) -> int: # -> type is a declaration
        return len(self.features)
    
    def __getitem__(self, index):
        image = Image.open(self.features[index]).convert("RGB")
        return self.transform(image), torch.tensor(self.labels[index], dtype=torch.float32)
    
    
def load_data_bananas(batch_size: int, edge_size: int = 256):
    train_ds = BananasDataset(is_train=True, edge_size=edge_size)
    val_ds = BananasDataset(is_train=False, edge_size=edge_size)
    train_iter = data.DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
    val_iter = data.DataLoader(val_ds, batch_size, shuffle=False, num_workers=4)
    return train_iter, val_iter