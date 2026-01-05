import torch
from torch import nn
from d2l import torch as d2l

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


def init_net(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# conv_arch is ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def create_net(conv_arch):
    conv_blocks = []
    in_channels = 1
    
    # convolution part
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        
    # linear part
    return nn.Sequential(
        *conv_blocks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))



def show_shape(conv_arch):
    X = torch.rand(size=(3, 1, 28, 28), dtype=torch.float32)
    net = create_net(conv_arch)
    for i, layer in enumerate(net):
        X = layer(X)
        print(f"{i}: {layer.__class__.__name__}\toutput shape: {X.shape}")


def train_epoch(net, updater, loss, train_iter, device):
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for X, y in train_iter:
        X, y = X.to(device), y.to(device)

        o_hat = net(X)
        mean_loss = loss(o_hat, y)
        updater.zero_grad()
        mean_loss.backward()
        updater.step()

        # calculate data
        total_samples += y.size(0)
        total_loss += y.size(0) * mean_loss.item()
        total_correct += (o_hat.argmax(dim=1) == y).sum().item()

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


def create_updater(net):
    return torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)


def create_loss():
    return nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean")

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]


device = decide_gpu_or_cpu()
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
net = create_net(small_conv_arch).to(device)
net.apply(init_net)
updater = create_updater(net)
loss = create_loss()
epochs = 10

with Timer("Training"):
    # train and return record
    train_losses, train_accs, test_losses, test_accs = train_whole(
        epochs, net, updater, loss, train_iter, test_iter, device
    )

plot_training_curves_single_figure(train_losses, train_accs, test_losses, test_accs)
