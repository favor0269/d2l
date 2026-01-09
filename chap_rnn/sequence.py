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


def create_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
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

T = 1000
time = torch.arange(0, T, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# Start background plotter to keep UI responsive and reuse the same figure
plotter = AsyncPlotter(time.numpy(), x.numpy(), title="Time Series and Prediction")

tau = 4
features = torch.zeros((T - tau, tau))

# from 0 to tau-1
# x: x_0 x_1 ... x_T
for i in range(tau):
    features[:, i] = x[i : T - tau + i]
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
net = create_net()
loss = create_loss()
updater = create_updater(net, lr)
num_epochs = 10

train_all(num_epochs, net, loss, updater, train_dataloader, test_dataloader)

predictions = predict(net, features)

# Add prediction curve to the existing figure without opening a new window
plotter.add_curve(time[tau:].numpy(), predictions.detach().cpu().numpy(), label="Predicted Values (y_hat)", color="orange")

multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))


plotter.add_curve(time[tau:].numpy(), multistep_preds[tau:].detach().cpu().numpy(), label="multistep_preditions", color="green")
