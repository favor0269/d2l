import os
import copy
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import pickle
from torch import nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "classify-leaves")


class Timer:

    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.name:
            print(f"[{self.name}] elapsed time: {self.elapsed:.2f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed:.2f} seconds")


# ================= Dataset =================
class LeafDataset(Dataset):
    def __init__(
        self,
        csv_file,
        data_dir,
        transform=None,
        train=True,
        le_path="label_encoder.pkl",
    ):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.le_path = le_path

        if train:
            self.le = LabelEncoder()
            self.data["label"] = self.le.fit_transform(self.data["label"])
            with open(self.le_path, "wb") as f:
                pickle.dump(self.le, f)
            print(f"[INFO] LabelEncoder saved to {self.le_path}")
        else:
            if os.path.exists(self.le_path):
                with open(self.le_path, "rb") as f:
                    self.le = pickle.load(f)
                print(f"[INFO] LabelEncoder loaded from {self.le_path}")
            else:
                self.le = None
                print(f"[WARNING] LabelEncoder not found, testing without labels")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.train:
            label = torch.tensor(row["label"], dtype=torch.long)
            return img, label
        else:
            return img


# ================= Transforms =================

transform = transforms.Compose(
    [
        # Randomly crop a portion of the image and resize it to 224x224
        # scale=(0.8, 1.0) means the crop size is randomly chosen between 80% and 100% of the original image
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # Randomly flip the image horizontally with a 50% probability
        transforms.RandomHorizontalFlip(),
        # Randomly flip the image vertically with a 50% probability
        transforms.RandomVerticalFlip(),
        # Randomly change brightness, contrast, saturation, and hue
        # brightness=0.2 → ±20% change
        # contrast=0.2   → ±20% change
        # saturation=0.2 → ±20% change
        # hue=0.1        → ±0.1 change in hue
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        # Convert PIL Image or numpy array to PyTorch tensor
        # Output shape: [C, H, W], pixel values scaled to [0.0, 1.0]
        transforms.ToTensor(),
        # Normalize each channel with mean=0.5 and std=0.5
        # Maps pixel values from [0, 1] to [-1, 1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

train_dataset = LeafDataset(
    csv_file=os.path.join(DATA_DIR, "train.csv"),
    data_dir=DATA_DIR,
    transform=transform,
    train=True,
)
test_dataset = LeafDataset(
    csv_file=os.path.join(DATA_DIR, "test.csv"),
    data_dir=DATA_DIR,
    transform=test_transform,
    train=False,
)


# ================= Network =================
class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.conv3 = (
            nn.Conv2d(input_channels, output_channels, 1, stride=strides)
            if use_1x1conv
            else None
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.nn.functional.relu(Y)


def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, output_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk


def create_net(num_classes):
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(
        b1,
        b2,
        b3,
        b4,
        b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return net


def init_net(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ================= Training / Evaluation =================
def train_epoch(net, optimizer, loss_fn, train_loader, device):
    net.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = net(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_samples += y.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(net, loss_fn, val_loader, device):
    net.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = net(X)
            loss = loss_fn(out, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_samples += y.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train_whole(epochs, net, optimizer, loss_fn, train_loader, val_loader, device):
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for epoch in range(epochs):
        tl, ta = train_epoch(net, optimizer, loss_fn, train_loader, device)
        vl, va = evaluate(net, loss_fn, val_loader, device)
        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        print(f"Epoch {epoch+1}: train_acc={ta:.4f}, val_acc={va:.4f}")
    return train_losses, train_accs, val_losses, val_accs


# ================= K-Fold Cross Validation =================
def k_fold_train(dataset, k=5, epochs=10, batch_size=256, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"--- Fold {fold+1}/{k} ---")
        train_subset = Subset(dataset, train_idx)
        # use lighter transforms for validation to avoid noisy estimates
        val_dataset = copy.copy(dataset)
        val_dataset.transform = test_transform
        val_subset = Subset(val_dataset, val_idx)
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=6
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=6
        )

        num_classes = len(dataset.le.classes_)
        net = create_net(num_classes).to(device)
        net.apply(init_net)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        train_whole(epochs, net, optimizer, loss_fn, train_loader, val_loader, device)

        # save acc
        vl, va = evaluate(net, loss_fn, val_loader, device)
        fold_results.append({"fold": fold + 1, "val_acc": va})

    avg_val_acc = sum(f["val_acc"] for f in fold_results) / k
    print(f"Average val accuracy: {avg_val_acc:.4f}")
    # return the last net
    return fold_results, net


# ================= Submission =================
def save_submission(
    net, test_dataset, label_encoder, device, save_path="submission.csv"
):
    net.eval()
    preds = []
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            out = net(X)
            pred_labels = out.argmax(1).cpu().numpy()
            pred_names = label_encoder.inverse_transform(pred_labels)
            preds.extend(pred_names)
    submission_df = pd.DataFrame(
        {"image": test_dataset.data["image"].tolist(), "label": preds}
    )
    submission_df.to_csv(save_path, index=False)
    print(f"[INFO] submission.csv saved to {save_path}")


# ================= Main =================
with Timer("Training"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results, best_net = k_fold_train(
        train_dataset, k=3, epochs=20, batch_size=256, device=device
    )
    save_submission(
        best_net, test_dataset, train_dataset.le, device, save_path="submission.csv"
    )
