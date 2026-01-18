import time
import math
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


def plot_heatmap(
    matrix,
    xlabel="Column",
    ylabel="Row",
    title="Heatmap",
    cmap="viridis",
    figsize=(8, 4),
    colorbar_label="Value",
    aspect="auto",
    save_path=None,
):
    """Plot a 2D heatmap from a numpy or torch array."""
    data = matrix.detach().cpu().numpy() if torch.is_tensor(matrix) else matrix
    # If attention has extra dims (e.g., batch, heads), average them out
    if data.ndim > 2:
        reduce_axes = tuple(range(data.ndim - 2))
        data = data.mean(axis=reduce_axes)
    plt.figure(figsize=figsize)
    im = plt.imshow(data, cmap=cmap, aspect=aspect)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_heatmaps_grid(
    matrix,
    max_images=8,
    cols=4,
    xlabel="Column",
    ylabel="Row",
    title_prefix="Heatmap",
    cmap="viridis",
    figsize=(10, 6),
    colorbar_label="Value",
    aspect="auto",
    save_path=None,
):
    """Plot multiple 2D heatmaps from higher-rank attention tensors.

    - matrix: tensor/array, e.g. (batch, heads, q_len, k_len)
    - max_images: number of slices to plot
    - cols: columns in the grid
    """

    data = matrix.detach().cpu().numpy() if torch.is_tensor(matrix) else matrix
    if data.ndim < 2:
        raise ValueError("matrix must have at least 2 dimensions")

    # Flatten leading dims into a list of 2D maps
    if data.ndim > 2:
        maps = data.reshape(-1, data.shape[-2], data.shape[-1])
    else:
        maps = data[None, ...]

    n = min(max_images, maps.shape[0])
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = numpy.array(axes).reshape(rows, cols)

    last_im = None
    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx < n:
            last_im = ax.imshow(maps[idx], cmap=cmap, aspect=aspect)
            ax.set_title(f"{title_prefix} {idx}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.axis("off")

    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            shrink=0.7,
            pad=0.02,
            label=colorbar_label,
            location="right",
        )
    # Avoid tight_layout warnings with colorbar; adjust spacing manually
    fig.subplots_adjust(wspace=0.4, hspace=0.5, right=0.9)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()