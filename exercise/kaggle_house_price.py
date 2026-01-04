import pandas

import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

import time


def decide_gpu_or_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def standardize_data(df):
    df_processed = df.copy()

    numeric_columns = df_processed.select_dtypes(
        include=["float64", "float32", "int64"]
    ).columns
    df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
    for col in numeric_columns:
        std = df_processed[col].std()
        if std > 0:
            df_processed[col] = (df_processed[col] - df_processed[col].mean()) / std
        else:
            df_processed[col] = 0

    categorical_columns = df_processed.select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        df_processed = pandas.get_dummies(
            df_processed,
            columns=categorical_columns,
            dummy_na=True,
        ).astype("float32")

    return df_processed


def init_net(m):
    if isinstance(m, nn.Linear):
        # Xavier initialize
        nn.init.xavier_uniform_(m.weight)  
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def create_net(input_dim):
    net = nn.Sequential(
        # this model is better complicated models i have tried
        nn.Linear(input_dim, 64),
        nn.Linear(64, 1),
    )
    net.apply(init_net)
    return net


def train_epoch(net, updater, loss, train_dataloader, device):
    total_loss = 0.0
    total_samples = 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()

        total_loss += l.item() * y.size(0)
        total_samples += y.size(0)
    return total_loss / total_samples


def create_updater(net):
    return torch.optim.Adam(net.parameters(), lr=0.002, weight_decay=1e-4)


def create_loss():
    return nn.MSELoss(reduction="mean")


def train_whole(num_epochs, net, updater, loss, train_dataloader, device):
    train_losses = []
    for i in range(num_epochs):
        train_loss = train_epoch(net, updater, loss, train_dataloader, device)
        train_losses.append(train_loss)
        print(f"epoch {i+1}")
        print(f"train_loss: {train_loss:.4f}")
    return train_losses


def plot_loss_curve(train_losses):
    # start from the sixth point
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(6, len(train_losses) + 1),
        train_losses[5:],
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()


def generate_kaggle_submission(net, test_dataloader, device):
    net.eval()
    with torch.no_grad():
        y_pred_list = []
        for X in test_dataloader:
            X = X[0].to(device)
            y_hat = net(X)
            y_pred_list.append(y_hat.cpu())
        y_pred = torch.cat(y_pred_list, dim=0)
        y_pred_orig = torch.expm1(y_pred).numpy()
        return y_pred_orig


def save_submission(y_pred, filename="submission.csv"):
    test_ids = pandas.read_csv(
        "d2l/data/home-data-for-ml-course/test.csv"
    )["Id"]
    submission = pandas.DataFrame({"Id": test_ids, "SalePrice": y_pred})
    submission.to_csv(filename, index=False)
    print("sucessfully create: submission.csv")


def preprocess_data(train_data, test_data):
    train_data = train_data.drop(columns=["Id"])
    test_data = test_data.drop(columns=["Id"])

    all_features = pandas.concat(
        [train_data.drop(columns=["SalePrice"]), test_data], axis=0
    )
    all_features = standardize_data(all_features).astype("float32")
    print("Shape of train + test: ", all_features.shape)

    # divide the all_features
    n_train = train_data.shape[0]
    X_train = all_features[:n_train].values
    X_test = all_features[n_train:].values

    # dont use (y-y.mean) / y.std , which will lead to nan output
    # however use log can safely standardlize y (SalePrice)
    y_train_tensor = torch.log1p(
        torch.tensor(train_data["SalePrice"].values, dtype=torch.float32)
    ).view(-1, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor


def create_dataloaders(
    x_train_tensor, y_train_tensor, x_test_tensor, batch_size=64, num_workers=4
):

    # train dataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # test dataLoader
    test_dataset = TensorDataset(x_test_tensor)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def get_k_fold_dataloaders(k, i, X_tensor, y_tensor, batch_size=64, num_worker=4):

    fold_size = X_tensor.shape[0] // k
    indices = torch.arange(X_tensor.shape[0])
    # get i th fold to be valid_loader
    valid_idx = indices[i * fold_size : (i + 1) * fold_size]
    # link other data to be train_loader 
    train_idx = torch.cat([indices[: i * fold_size], indices[(i + 1) * fold_size :]])

    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    valid_dataset = TensorDataset(X_tensor[valid_idx], y_tensor[valid_idx])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker
    )
    return train_loader, valid_loader


def k_fold_train(X_tensor, y_tensor, k=5, num_epochs=100, device=None, num_workers=4):
    train_loss_sum, valid_loss_sum = 0, 0

    for i in range(k):
        train_loader, valid_loader = get_k_fold_dataloaders(
            k, i, X_tensor, y_tensor, batch_size=64, num_worker=num_workers
        )
        # re generate net updater loss_fn every fold (independency)
        net = create_net(X_tensor.shape[1]).to(device)
        updater = create_updater(net)
        loss_fn = create_loss()

        # train
        train_losses = []
        for _ in range(num_epochs):
            train_loss = train_epoch(net, updater, loss_fn, train_loader, device)
            train_losses.append(train_loss)

        # calculate loss
        net.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for X_val, y_val in valid_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_hat = net(X_val)
                l = loss_fn(y_hat, y_val)
                total_loss += l.item() * y_val.size(0)
                total_samples += y_val.size(0)
            valid_loss = total_loss / total_samples

        # the last train loss in an epoch
        train_loss_sum += train_losses[-1]
        valid_loss_sum += valid_loss
        print(
            f"fold {i+1}: train_loss={train_losses[-1]:.4f}, valid_loss={valid_loss:.4f}"
        )

    print(
        f"k fold mean: train_loss: {train_loss_sum/k:.4f}, valid_loss: {valid_loss_sum/k:.4f}"
    )
    return train_loss_sum / k, valid_loss_sum / k


batch_size = 64
num_workers = 4
device = decide_gpu_or_cpu()
num_epochs = 50

train_data = pandas.read_csv(
    "d2l/data/home-data-for-ml-course/train.csv"
)
test_data = pandas.read_csv(
    "d2l/data/home-data-for-ml-course/test.csv"
)

X_train_tensor, y_train_tensor, X_test_tensor = preprocess_data(train_data, test_data)

# 1.use k fold train to evaluate
k_fold_train(X_train_tensor, y_train_tensor, k=5, num_epochs=num_epochs, device=device)

# 2.use all data to train final model
train_dataloader, test_dataloader = create_dataloaders(
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    batch_size=batch_size,
    num_workers=num_workers,
)

net = create_net(X_train_tensor.shape[1]).to(device)
updater = create_updater(net)
loss = create_loss()
train_losses = train_whole(num_epochs, net, updater, loss, train_dataloader, device)
plot_loss_curve(train_losses)

# 3.generate final data
y_pred = generate_kaggle_submission(net, test_dataloader, device)
y_pred = y_pred.flatten()

save_submission(y_pred)
