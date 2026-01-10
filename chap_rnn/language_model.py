import text_preprocess
import torch
import random
import matplotlib.pyplot as plt


def plot(
    x,
    y=None,
    xlabel=None,
    ylabel=None,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ylim=None,
    legend=None,
    figsize=(6, 4),
):
    """Lightweight drop-in replacement for d2l.plot with common args."""

    plt.figure(figsize=figsize)

    # Normalize inputs to lists of curves
    x_list = (
        x
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple))
        else [x]
    )
    if y is None:
        y_list = [
            curve if isinstance(curve, (list, tuple)) else list(curve)
            for curve in x_list
        ]
        x_list = [list(range(1, len(curve) + 1)) for curve in y_list]
    else:
        y_list = (
            y
            if isinstance(y, (list, tuple)) and y and isinstance(y[0], (list, tuple))
            else [y]
        )
        if not isinstance(x, (list, tuple)) or (
            x and not isinstance(x[0], (list, tuple))
        ):
            x_list = [x] * len(y_list)

    for xs, ys in zip(x_list, y_list):
        plt.plot(xs, ys)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if legend:
        plt.legend(legend)
    plt.tight_layout()
    plt.show()


def plot_token_frequency(vocab, topk=None):

    freqs = [freq for _, freq in vocab.token_freqs]
    if topk:
        freqs = freqs[:topk]
    plot(
        freqs,
        xlabel="token: x",
        ylabel="frequency: n(x)",
        xscale="log",
        yscale="log",
    )


def plot_ngram_frequencies(vocabs, labels=None, topk=None):

    freq_lists = []
    for vocab in vocabs:
        freqs = [freq for _, freq in vocab.token_freqs]
        if topk:
            freqs = freqs[:topk]
        freq_lists.append(freqs)
    plot(
        freq_lists,
        xlabel="token: x",
        ylabel="frequency: n(x)",
        xscale="log",
        yscale="log",
        legend=labels,
    )


# unique corpus
tokens = text_preprocess.tokenize(text_preprocess.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = text_preprocess.Vocabulary(corpus)

print("unique corpus:\n", vocab.token_freqs[:10], "\n")


# bigram corpus
bigram_corpus = list(zip(corpus[:-1], corpus[1:]))
bigram_vocab = text_preprocess.Vocabulary(bigram_corpus)

print("bigram corpus:\n", bigram_vocab.token_freqs[:10], "\n")


# trigram corpus
trigram_corpus = list(zip(corpus[:-2], corpus[1:-1], corpus[2:]))
trigram_vocab = text_preprocess.Vocabulary(trigram_corpus)

print("trigramcorpus:\n", trigram_vocab.token_freqs[:10], "\n")

plot_ngram_frequencies(
    [vocab, bigram_vocab, trigram_vocab],
    labels=["unigram", "bigram", "trigram"],
)


