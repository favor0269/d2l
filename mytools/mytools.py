import time
import torch
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class TimedBlock:

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


def decide_gpu_or_cpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return numpy.array(self.times).cumsum().tolist()


def plot_lines(
    x,
    y=None,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ylim=None,
    figsize=(6, 4),
    grid=True,
    save_path=None,
    linewidth=2.0,
    title_size=16,
    label_size=14,
    tick_size=12,
    max_xticks=None,
    max_yticks=None,
):
    """Plot one or many curves on the same axes with optional axis settings."""

    plt.figure(figsize=figsize)

    # Normalize inputs to lists of curves
    x_list = (
        x if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)) else [x]
    )
    if y is None:
        y_list = [curve if isinstance(curve, (list, tuple)) else list(curve) for curve in x_list]
        x_list = [list(range(len(curve))) for curve in y_list]
    else:
        y_list = (
            y if isinstance(y, (list, tuple)) and y and isinstance(y[0], (list, tuple)) else [y]
        )
        if not isinstance(x, (list, tuple)) or (x and not isinstance(x[0], (list, tuple))):
            x_list = [x] * len(y_list)

    ax = plt.gca()
    for xs, ys in zip(x_list, y_list):
        ax.plot(xs, ys, linewidth=linewidth)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size)
    if title:
        ax.set_title(title, fontsize=title_size)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(legend)
    ax.tick_params(axis="both", labelsize=tick_size)
    if max_xticks:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
    if max_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))
    if grid:
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()