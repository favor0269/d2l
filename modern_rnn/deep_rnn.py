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
from chap_rnn import RNN_concise
from chap_rnn import RNN


batch_size = 32
num_steps = 35
train_iter, vocab = seq_dataset.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, mytools.decide_gpu_or_cpu()
num_epochs, lr = 500, 1


num_inputs = vocab_size
num_layers = 2
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
net = RNN_concise.RNNModel(lstm_layer, len(vocab))
net = net.to(device)
total_perplexities = RNN.train_all(net, train_iter, vocab, lr, num_epochs, device)
mytools.plot_lines(total_perplexities)
