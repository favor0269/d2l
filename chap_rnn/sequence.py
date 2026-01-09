import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn
import threading
import queue


class AsyncPlotter:
    """Background plotter to avoid blocking the main thread."""

    def __init__(self, time, series, title="Time Series"):
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, args=(time, series, title), daemon=False
        )
        self._thread.start()

    def add_curve(self, x, y, label, color=None):
        # Queue a new curve to be drawn by the background thread
        self._queue.put((x, y, label, color))

    def _worker(self, time, series, title):
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.canvas.mpl_connect("close_event", lambda event: self._stop.set())
        ax.plot(time, series, label="Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        fig.canvas.draw()
        fig.show()

        while plt.fignum_exists(fig.number) and not self._stop.is_set():
            try:
                x, y, label, color = self._queue.get(timeout=0.1)
                # Guard against shape mismatch to avoid crashing the thread
                if len(x) != len(y):
                    continue
                ax.plot(x, y, label=label, color=color)
                ax.legend()
                fig.canvas.draw()
                plt.pause(0.001)
            except queue.Empty:
                plt.pause(0.001)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


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

# Start background plotter to keep UI responsive and reuse the same figure
plotter = AsyncPlotter(time.numpy(), x.numpy(), title="Time Series and Prediction")

# Sliding window length tau, predict k_step-ahead
tau = 4
k_step = 64  # training is always 1-step; k_step only affects iterative forecast below
feature_len = T - tau
features = torch.zeros((feature_len, tau))

# build features with sliding window
for i in range(tau):
    features[:, i] = x[i : i + feature_len]
labels = x[tau:].reshape((-1, 1))


print(features.shape)
print(labels.shape)

batch_size = 16
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
plotter.add_curve(
    time_slice_all.numpy(),
    predictions_all.detach().cpu().numpy(),
    label=f"{k_step}-step iterative prediction (all)",
    color="orange",
)

# Optional autoregressive demo: roll forward from training end using model outputs
MULTI_STEP_DEMO = False
FUTURE_STEPS = 100  # how many points to roll out
if MULTI_STEP_DEMO:
    start_idx = n_train
    steps = T - (start_idx + tau)
    steps = min(steps, FUTURE_STEPS)
    multistep_preds = torch.zeros(tau + steps)
    multistep_preds[:tau] = x[start_idx : start_idx + tau]
    for i in range(tau, tau + steps):
        multistep_preds[i] = net(multistep_preds[i - tau : i].reshape((1, -1)))

    future_time = time[start_idx + tau : start_idx + tau + steps]
    plotter.add_curve(
        future_time.numpy(),
        multistep_preds[tau:].detach().cpu().numpy(),
        label=f"autoregressive roll-out (from {start_idx}, k={k_step})",
        color="green",
    )
