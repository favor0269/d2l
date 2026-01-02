import torch
from torch import nn

from torch.utils import data

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

import time


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


def get_dataloader_workers():
    return 4


def load_data_fashion_mnist(batch_size, resize=None):

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            mnist_train,
            batch_size,
            shuffle=True,
            num_workers=get_dataloader_workers(),
            pin_memory=True,
        ),
        data.DataLoader(
            mnist_test,
            batch_size,
            shuffle=False,
            num_workers=get_dataloader_workers(),
            pin_memory=True,
        ),
    )


def init_net(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.normal_(m.bias, mean=0, std=0.01)


def create_net():
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(p=0.05),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(p=0.05),
        nn.Linear(64, 10),
    )
    net.apply(init_net)
    return net


def create_updater(net):
    return torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


def create_cross_entropy_loss():
    return nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean")


def train_epoch(net, updater, loss, train_iter, device):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)

        o_hat = net(X)
        l = loss(o_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()

        # calculate data
        total_loss += l.item() * y.size(0)
        total_correct += (o_hat.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate_by_test(net, loss, test_iter, device):
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)

            o_hat = net(X)
            l = loss(o_hat, y)

            # calculate data
            total_loss += l.item() * y.size(0)
            total_correct += (o_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

        return total_loss / total_samples, total_correct / total_samples


def train_whole(epochs, net, updater, loss, train_iter, test_iter, device):
    # local
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for i in range(epochs):
        train_loss, train_accuracy = train_epoch(net, updater, loss, train_iter, device)
        test_loss, test_accuracy = evaluate_by_test(net, loss, test_iter, device)

        # record data
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

        print(f"epoch {i+1}")
        print(f"train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}")
        print(f"test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.4f}")

    return train_losses, train_accs, test_losses, test_accs


# define constant
FIGSIZE = (8, 5)

LOSS_COLOR_TRAIN = "tab:red"
LOSS_COLOR_TEST = "tab:orange"
ACC_COLOR_TRAIN = "tab:blue"
ACC_COLOR_TEST = "tab:cyan"

MARKER_LOSS = "o"
MARKER_ACC = "x"

TITLE = "Training and Testing Loss & Accuracy"
X_LABEL = "Epoch"
LOSS_Y_LABEL = "Loss"
ACC_Y_LABEL = "Accuracy"
LEGEND_LOC = "center right"


def plot_training_curves_single_figure(
    train_losses, train_accs, test_losses, test_accs
):
    epochs = range(1, len(train_losses) + 1)

    _, ax1 = plt.subplots(figsize=FIGSIZE)

    # left y axis：Loss
    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel(LOSS_Y_LABEL, color=LOSS_COLOR_TRAIN)
    ax1.plot(
        epochs,
        train_losses,
        label="Train Loss",
        color=LOSS_COLOR_TRAIN,
        marker=MARKER_LOSS,
    )
    ax1.plot(
        epochs,
        test_losses,
        label="Test Loss",
        color=LOSS_COLOR_TEST,
        marker=MARKER_LOSS,
    )
    ax1.tick_params(axis="y", labelcolor=LOSS_COLOR_TRAIN)
    ax1.grid(True)

    # right y axis：Accuracy
    # share x axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(ACC_Y_LABEL, color=ACC_COLOR_TRAIN)
    ax2.plot(
        epochs, train_accs, label="Train Acc", color=ACC_COLOR_TRAIN, marker=MARKER_ACC
    )
    ax2.plot(
        epochs, test_accs, label="Test Acc", color=ACC_COLOR_TEST, marker=MARKER_ACC
    )
    ax2.tick_params(axis="y", labelcolor=ACC_COLOR_TRAIN)

    # compose the legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=LEGEND_LOC)

    plt.title(TITLE)
    plt.tight_layout()
    plt.show()


def decide_gpu_or_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


device = decide_gpu_or_cpu()
net = create_net().to(device)
updater = torch.optim.SGD(net.parameters(), lr=0.15, momentum=0.9, weight_decay=1e-4)
train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
loss = create_cross_entropy_loss()
epochs = 30

with Timer("Training"):
    # train and return record
    train_losses, train_accs, test_losses, test_accs = train_whole(
        epochs, net, updater, loss, train_iter, test_iter, device
    )

# draw plot
plot_training_curves_single_figure(train_losses, train_accs, test_losses, test_accs)
