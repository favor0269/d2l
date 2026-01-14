import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import sys

# Make project root importable so sibling packages resolve
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from chap_rnn import seq_dataset, text_preprocess


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current file dir
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "fra-eng")  # dataset root


def read_data_nmt():
    """Load raw parallel corpus text."""
    data_dir = DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR not found: {data_dir}. Please place the banana-detection dataset there."
        )

    txt_path = os.path.join(data_dir, "fra.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()
    return lines


def preprocess_nmt(text):
    """Normalize spaces and lowercase; ensure punctuation is space-prefixed."""

    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()

    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)


def tokenize_nmt(text, num_examples=None):
    """Split raw text into token lists for source/target."""
    source, target = [], []
    for i, line in enumerate(text.split("\n")):
        if num_examples and i > num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))  # English tokens
            target.append(parts[1].split(" "))  # French tokens
    return source, target


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):

    len_x = [len(l) for l in xlist]  # lengths of source sequences
    len_y = [len(l) for l in ylist]  # lengths of target sequences
    fig, ax = plt.subplots()  # new figure/axes
    n, bins, patches = ax.hist([len_x, len_y], label=legend)  # stacked hist
    # set x, y axis title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for patch in patches[1].patches:  # visually distinguish second series
        patch.set_hatch("/")

    # display Legend (image example)
    ax.legend()
    return fig, ax


# padding or truncation facilitates matrix operations.
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab["<eos>"]] for line in lines]
    array = torch.tensor(
        [truncate_pad(line, num_steps, vocab["<pad>"]) for line in lines]
    )
    # get not <pad> numbers include <eos> by rows
    valid_len = (array != vocab["<pad>"]).type(torch.float32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Build DataLoader for NMT pairs with padding/truncation."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = text_preprocess.Vocabulary(
        source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )
    tgt_vocab = text_preprocess.Vocabulary(
        target, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # Tensors before batching:
    #   src_array: (num_samples, num_steps) int64
    #   src_valid_len: (num_samples,) float32
    #   tgt_array: (num_samples, num_steps) int64
    #   tgt_valid_len: (num_samples,) float32
    # After DataLoader batching:
    #   X: (batch_size, num_steps)
    #   X_valid_len: (batch_size,)
    #   Y: (batch_size, num_steps)
    #   Y_valid_len: (batch_size,)
    dataset = TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab


def test1():
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    source, target = tokenize_nmt(text)
    print(source[:6])  # peek first few tokenized source sentences

    fig, ax = show_list_len_pair_hist(
        legend=["xlist", "ylist"],
        xlabel="Sequence Length",
        ylabel="Count",
        xlist=source,
        ylist=target,
    )

    src_vocab = text_preprocess.Vocabulary(
        source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )

    print(len(src_vocab))

    cc = truncate_pad(src_vocab[source[5000]], 10, src_vocab["<pad>"])
    print(cc)
    cc = src_vocab.to_tokens(cc)
    print(cc)


def test2():
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print("X:", X.type(torch.int32))
        print("X valid len:", X_valid_len)
        print("Y:", Y.type(torch.int32))
        print("Y valid len:", Y_valid_len)
        break
