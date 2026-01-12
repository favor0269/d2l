import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math
import torch
from torch import nn
from torch.nn import functional as F
from chap_rnn import seq_dataset
from mytools import mytools
from chap_rnn import RNN


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device),
        )

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [
        W_xi,
        W_hi,
        b_i,
        W_xf,
        W_hf,
        b_f,
        W_xo,
        W_ho,
        b_o,
        W_xc,
        W_hc,
        b_c,
        W_hq,
        b_q,
    ]
    for param in params:
        param.requires_grad_(True)

    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (
        torch.zeros((batch_size, num_hiddens), device=device),
        torch.zeros((batch_size, num_hiddens), device=device),
    )


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = (
        params
    )
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


batch_size = 32
num_steps = 35
train_iter, vocab = seq_dataset.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, mytools.decide_gpu_or_cpu()
num_epochs, lr = 500, 1
net = RNN.RNNModelScratch(
    len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm
)

total_perplexities = RNN.train_all(net, train_iter, vocab, lr, num_epochs, device)
mytools.plot_lines(total_perplexities)
