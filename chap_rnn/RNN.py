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

from . import seq_dataset
from . import text_preprocess
from mytools import mytools


# =====================nn.RNN scratch=========================
def get_params(vocab_size, num_hiddens, device):

    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state  # H: (batch_size, num_hiddens)
    outputs = []

    for X in inputs:  # iterate by time; X: (batch_size, vocab_size)
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q  # Y: (batch_size, vocab_size)
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


# ============================================================
class RNNModelScratch:

    def __init__(
        self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn
    ):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # the whole net shares W_xh, W_hh, b_h, W_hq, b_q
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # (batch_size, num_steps) -> (num_steps, batch_size, vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # indices list
    # get_input only take last output, because we have H_t
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1)
    )  # (1,1)
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
        # we dont care y_hat here because we have real y
        # teacher forcing

    # here the y we used are
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    # transform idx to token and return
    return "".join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    # Gradient clipping: compute global L2 norm over all params, then rescale
    # grads so that ||g|| <= theta. This prevents exploding gradients.

    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    timer = mytools.Timer()
    metric = mytools.Accumulator(2)
    for X, Y in train_iter:
        # shape:
        # X.shape = (batch_size, num_steps)
        # Y.shape = (batch_size, num_steps)
        if state is None or use_random_iter:
            # discontinuity in the sequence of events
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # computational graph separation
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)  # y: (t*b,)
        # reshape walkthrough (b=batch, t=time):
        # Y       shape (b, t): [[b0t0, b0t1, b0t2],
        #                        [b1t0, b1t1, b1t2]]
        # Y.T     shape (t, b): [[b0t0, b1t0],
        #                        [b0t1, b1t1],
        #                        [b0t2, b1t2]]
        # reshape -> (t*b,):    [b0t0, b1t0, b0t1, b1t1, b0t2, b1t2]
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)  # y_hat: (t*b, vocab_size)
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)

            updater(batch_size=1)

        metric.add((l * y.numel()).item(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_all(
    net,
    train_iter,
    vocab,
    lr,
    num_epochs,
    device,
    use_random_iter=False,
    predict_prefix_text="time traveller",
    num_preds=50,
):
    total_perplexities = []

    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    def predict_prefix_local(num_preds_override=None):
        n = num_preds if num_preds_override is None else num_preds_override
        return predict(predict_prefix_text, n, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, updater, device, use_random_iter
        )
        total_perplexities.append(ppl)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:4d}, ppl: {ppl:3.4f}")
            print(predict_prefix_local())

    print(f"perplexity {ppl:.1f}, {speed:.1f} corpus/s {str(device)}")
    print(predict_prefix_local())
    print(predict_prefix_local())

    return total_perplexities


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = seq_dataset.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 512
    device = mytools.decide_gpu_or_cpu()

    net = RNNModelScratch(
        len(vocab),
        num_hiddens,
        device,
        get_params,
        init_rnn_state,
        rnn,
    )

    print(predict("time traveller ", 10, net, vocab, device))

    num_epochs, lr = 500, 1
    train_all(net, train_iter, vocab, lr, num_epochs, device)


if __name__ == "__main__":
    main()


# TimeMachine 原始文本
#    ↓
# 字符 → vocab 编号
#    ↓
# X, Y from train_iter
# X.shape = (batch_size, num_steps)
# Y.shape = (batch_size, num_steps)

#    ↓  (net(X, state))
# X.T → one-hot
# inputs.shape = (num_steps, batch_size, vocab_size)

#    ↓  (rnn 按时间步循环)
# for t in num_steps:
#     X_t: (batch_size, vocab_size)
#     H:   (batch_size, num_hiddens)
#     Y_t: (batch_size, vocab_size)

#    ↓
# outputs cat
# y_hat.shape = (batch_size * num_steps, vocab_size)

#    ↓
# Y.T.reshape(-1)
# y.shape = (batch_size * num_steps)

#    ↓
# CrossEntropyLoss


# ============================================================
# End-to-end walkthrough (shapes + flow) with the current code
# Example: prefix = "time traveller " and num_preds = 10
#
# 1) Data prep (see seq_dataset.load_data_time_machine)
#    - raw text -> list of chars -> vocab (char to idx)
#    - build corpus indices and slice into mini-batches
#    - batch_size = 32, num_steps = 35 by default here
#    - train_iter yields (X, Y) where:
#        X.shape = (batch_size, num_steps)
#        Y.shape = (batch_size, num_steps)
#
# 2) Model (RNNModelScratch)
#    - one-hot inside __call__: X.T -> (num_steps, batch_size, vocab_size)
#    - forward rnn loop:
#        per time step t:
#           X_t.shape = (batch_size, vocab_size)
#           H.shape   = (batch_size, num_hiddens)
#           Y_t.shape = (batch_size, vocab_size)
#    - outputs concatenated over time:
#        y_hat.shape = (batch_size * num_steps, vocab_size)
#    - state returned: tuple with H of shape (batch_size, num_hiddens)
#
# 3) Loss
#    - Y.T.reshape(-1) -> y.shape = (batch_size * num_steps)
#    - CrossEntropyLoss compares y_hat vs y (class indices)
#
# 4) Training loop (train_all → train_epoch)
#    - for each batch: forward -> loss -> backward -> grad clip -> update
#    - metric accumulates total loss and token count
#    - returns perplexity = exp(total_loss/total_tokens)
#
# 5) Prediction (predict)
#    - init state with batch_size=1
#    - seed outputs with first char of prefix
#    - iterate prefix[1:], feeding previous real char; state rolls forward
#    - generation loop num_preds times:
#         feed last predicted char -> rnn -> take argmax over vocab
#    - final outputs list are vocab indices -> join idx_to_token -> string
#
# Shapes during predict for prefix "time traveller " (len=15) and num_preds=10:
#    state: (1, num_hiddens)
#    get_input() gives tensor[[last_idx]] with shape (1,1)
#    one-hot inside net: (1,1) -> (1, vocab_size) then viewed as (1,1,vocab_size)
#    rnn single step outputs one Y_t: (1, vocab_size); state keeps shape (1, num_hiddens)
#    loop repeats for each prefix char, then for each predicted step
#    result length = len(prefix) + num_preds = 25 chars
#
# 6) Quick manual run (CPU/GPU auto-decided by mytools.decide_gpu_or_cpu):
#    python chap_rnn/RNN.py
#    - prints a short sampled string before training
#    - trains for num_epochs=500 with lr=1
#    - every 10 epochs prints a sample starting with "time traveller"
#    - final perplexity + two samples ("time traveller" and "traveller")
