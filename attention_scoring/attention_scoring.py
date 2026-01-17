import os
import sys
import torch
from torch import nn
import math
from d2l import torch as d2l

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def sequence_mask(X, valid_len, value=0):

    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value

    # 1. torch.arange(maxlen) -> tensor([0,1,2,3...maxlen-1])
    # 2. [None, :] -> (1, maxlen)
    # 3. valid_len[:, None] -> (len, 1)
    # 4. broadcast like:
    # [[0,1,2,3,4] < 3] -> [True, True, True, False, False]
    # [[0,1,2,3,4] < 2] -> [True, True, False, False, False]
    # [[0,1,2,3,4] < 4] -> [True, True, True, True, False]
    # 5. invert ~mask
    # 6. False -> 0
    return X


def masked_softmax(X, valid_lens):
    # X: 3D tesnor
    # valide_lens: None or 1D or 2D
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        # key_size   query_size  num_hiddens：
        # K          Q           H
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # Flowchart:
        # queries (input)          : (B, Q, d_q)
        # keys (input)             : (B, K, d_k)
        # values (input)           : (B, K, V)

        # queries (mapped)        : (B, Q, H)
        # keys (mapped)           : (B, K, H)

        # features                : (B, Q, K, H)
        # -> (B, Q, K, 1) and squeeze to (B, Q, K)
        # scores                  : (B, Q, K)
        # attention_weights       : (B, Q, K)

        # output                  : (B, Q, V)
        # features =
        #     [
        #     [
        #         [ [2, 1], [2, 0], [1, 1] ],   # q1 + k1,k2,k3
        #         [ [1, 1], [2, 1], [0, 2] ]    # q2 + k1,k2,k3
        #     ]
        #     ]
        # shape = (1, 2, 3, 2)
        # scores =
        #     [
        #     [
        #         [3, 2, 2],   # q1 for k1,k2,k3
        #         [2, 3, 2]    # q2 for k1,k2,k3
        #     ]
        #     ]
        # shape = (1, 2, 3)
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # attention_weights =
        #     [
        #     [
        #         [0.576, 0.212, 0.212],   # q1
        #         [0.212, 0.576, 0.212]    # q2
        #     ]
        #     ]
        # shape = (1, 2, 3) # (B, Q, K)
        # values =
        #     [
        #     [   # batch 0
        #         [10, 0],   # v1
        #         [0, 10],   # v2
        #         [5, 5]     # v3
        #     ]
        #     ]
        # shape = (1, 3, 2)  # (B, K, V)

        # finally return (B, Q, V)
        return torch.bmm(self.dropout(self.attention_weights), values)


def test():
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))

    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))

    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))


class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape:
    # queries (batch_size, queries_size, d)
    # keys (batch_size, keys_size, d)
    # values (batch_size, keys_size, d)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # output1 = w11*v1 + w12*v2 + w13*v3 + w14*v4
        # = [ w11*v11 + w12*v21 + w13*v31 + w14*v41,
        #     w11*v12 + w12*v22 + w13*v32 + w14*v42 ]
