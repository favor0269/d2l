import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn


def plot_curves(curves_dict, title="Time Series and Predictions"):
    """Plot original series and multiple predicted curves on the same figure.

    Args:
        curves_dict: dict with special keys:
                     - "time": main time axis (required)
                     - "original": original series (required)
                     - other keys: (time, values) tuples or just values for prediction curves
        title: figure title
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    time = curves_dict.pop("time", None)
    true_series = curves_dict.pop("original", None)

    if time is None or true_series is None:
        raise ValueError("curves_dict must contain 'time' and 'original' keys")

    # convert to numpy if needed
    if isinstance(time, torch.Tensor):
        time = time.numpy()
    if isinstance(true_series, torch.Tensor):
        true_series = true_series.numpy()

    ax.plot(time, true_series, label="Time Series", linewidth=2)

    # plot all prediction curves
    for label, data in curves_dict.items():
        # check if data is tuple (x_time, y_values) or just y_values
        if isinstance(data, (tuple, list)) and len(data) == 2:
            x, y = data
        else:
            # just y_values, use main time axis
            x = None
            y = data

        # convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # use main time axis if x not provided
        if x is None:
            x = time[: len(y)]

        ax.plot(x, y, label=label, alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.canvas.draw()
    plt.show()


def init_net(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def create_net(input_dim=4):
    net = nn.Sequential(nn.Linear(input_dim, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_net)
    return net


def create_loss():
    return nn.MSELoss(reduction="mean")


def create_updater(net, lr):
    return torch.optim.Adam(net.parameters(), lr, weight_decay=0)


def train_epoch(net, loss, updater, dataloader):
    net.train()
    total_loss = 0.0
    total_samples = 0
    for X, y in dataloader:
        y_hat = net(X)
        updater.zero_grad()
        mean_loss = loss(y_hat, y)
        mean_loss.backward()
        updater.step()

        total_samples += y.size(0)
        total_loss += y.size(0) * mean_loss.item()
    return total_loss / total_samples


def evaluate_loss(net, loss, dataloader):
    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_samples = 0
        for X, y in dataloader:
            y_hat = net(X)
            mean_loss = loss(y_hat, y)

            total_samples += y.size(0)
            total_loss += y.size(0) * mean_loss.item()
        return total_loss / total_samples


def train_all(num_epochs, net, loss, updater, train_dataloader, test_dataloader):
    train_losses, test_losses = [], []
    for i in range(num_epochs):
        train_loss = train_epoch(net, loss, updater, train_dataloader)
        test_loss = evaluate_loss(net, loss, test_dataloader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"epoch {i+1}")
        print(f"train_loss: {train_loss:8.4f}, test_loss: {test_loss:8.4f}")


def predict(net, features):
    net.eval()
    with torch.no_grad():
        predictions = net(features)
    return predictions


def k_step_predict_iterative(net, series, tau, k_step):
    """Use a 1-step model to iteratively predict k steps ahead for each position.

    Returns a tensor of length len(series) - tau - k_step + 1, aligned to time
    indices time[tau + k_step - 1 :].
    """
    device = next(net.parameters()).device
    series = series.to(device)
    preds = []
    with torch.no_grad():
        max_start = len(series) - tau - k_step + 1
        for start in range(max_start):
            window = series[start : start + tau].reshape(1, -1)
            cur = window
            pred = None
            for _ in range(k_step):
                pred = net(cur)
                cur = torch.cat([cur[:, 1:], pred], dim=1)
            preds.append(pred.squeeze(1))
    if len(preds) == 0:
        return torch.empty(0)
    return torch.cat(preds)


T = 1000
time = torch.arange(0, T, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# Sliding window length tau, predict k_step-ahead
tau = 4
k_step = 16  # training is always 1-step; k_step only affects iterative forecast below
feature_len = T - tau
features = torch.zeros((feature_len, tau))

# build features with sliding window
for i in range(tau):
    features[:, i] = x[i : i + feature_len]
labels = x[tau:].reshape((-1, 1))


batch_size = 32
n_train = 600
num_workers = 4

train_dataset = data.TensorDataset(features[:n_train], labels[:n_train])
train_dataloader = data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataset = data.TensorDataset(features[n_train:], labels[n_train:])
test_dataloader = data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

lr = 0.05
net = create_net(tau)
loss = create_loss()
updater = create_updater(net, lr)
num_epochs = 10

train_all(num_epochs, net, loss, updater, train_dataloader, test_dataloader)

# 1) one-pass k-step prediction for the whole series (uses true windows)
predictions_all = k_step_predict_iterative(net, x, tau, k_step)
time_slice_all = time[tau + k_step - 1 : tau + k_step - 1 + len(predictions_all)]

# 2) Optional autoregressive demo: roll forward from training end using model outputs
curves_dict = {
    "time": time,
    "original": x,
}
curves_dict[f"{k_step}-step iterative prediction (all)"] = (
    time_slice_all,
    predictions_all,
)

MULTI_STEP_DEMO = True
FUTURE_STEPS = 50  # how many points to roll out
if MULTI_STEP_DEMO:
    start_idx = n_train
    steps = T - (start_idx + tau)
    steps = min(steps, FUTURE_STEPS)
    multistep_preds = torch.zeros(tau + steps)
    multistep_preds[:tau] = x[start_idx : start_idx + tau]
    for i in range(tau, tau + steps):
        multistep_preds[i] = net(multistep_preds[i - tau : i].reshape((1, -1)))

    future_time = time[start_idx + tau : start_idx + tau + steps]
    curves_dict[f"autoregressive roll-out (from {start_idx}, k={k_step})"] = (
        future_time,
        multistep_preds[tau:],
    )

# Plot all curves at the end
plot_curves(curves_dict, title="Time Series and Predictions")
