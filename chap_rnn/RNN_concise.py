import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
import os
from d2l import torch as d2l

# Ensure project root is on path so sibling mytools package is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import seq_dataset
import text_preprocess
from mytools import mytools
import RNN


class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size, *args, **kwargs):
        super(RNNModel, self).__init__(*args, **kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        # RNN could be bidirectional or unidirectional
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, H_t):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32).to(next(self.parameters()).device)
        H, state = self.rnn(X, H_t)
        # H shape: (num_steps, batch_size, num_hiddens)
        # H_t shape: (1, batch_size, num_hiddens)
        # H_t.squeeze(0) = H[-1, :, :] (batch_size, num_hiddens)
        # we reshape it as (num_steps * batch_size, num_hiddens)
        # which is convenient for calculate loss
        output = self.linear(H.reshape((-1, H.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU regards tensor as hidden states
            return torch.zeros(
                (
                    self.num_directions * self.rnn.num_layers,
                    batch_size,
                    self.num_hiddens,
                ),
                device=device,
            )
        else:
            # nn.LSTM regards tuple as hidden states
            return (
                torch.zeros(
                    (
                        self.num_directions * self.rnn.num_layers,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
                torch.zeros(
                    (
                        self.num_directions * self.rnn.num_layers,
                        batch_size,
                        self.num_hiddens,
                    ),
                    device=device,
                ),
            )


batch_size, num_steps = 32, 35
train_iter, vocab = seq_dataset.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256

device = mytools.decide_gpu_or_cpu()
rnn_layer = nn.RNN(len(vocab), num_hiddens).to(device)
net = RNNModel(rnn_layer, vocab_size=len(vocab)).to(device)

num_epochs = 500
lr = 1
total_perplexities = RNN.train_all(net, train_iter, vocab, lr, num_epochs, device)

mytools.plot_lines(total_perplexities)