import math
import torch
from torch import nn
from d2l import torch as d2l


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
    # 6. False -> value(default=0)
    return X


def masked_softmax(X, valid_lens):
    # X: 3D tesnor
    # valide_lens: None or 1D or 2D
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:  # :valid_lens = batch_size
            valid_lens = torch.repeat_interleave(
                valid_lens, shape[1]
            )  # repeat time_steps times
        else:
            valid_lens = valid_lens.reshape(-1)  # one-to-one correspondence

        # transform to 2D to calculate softmax
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shapes when used inside multi-head attention (after transpose_qkv):
    # queries (batch_size*num_heads, num_queries, head_dim)
    # keys    (batch_size*num_heads, num_keys, head_dim)
    # values  (batch_size*num_heads, num_keys, value_dim_per_head)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # output1 = w11*v1 + w12*v2 + w13*v3 + w14*v4
        # = [ w11*v11 + w12*v21 + w13*v31 + w14*v41,
        #     w11*v12 + w12*v22 + w13*v32 + w14*v42 ]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size,
        key_size,
        value_size,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # For parallel computing
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # regardless 1D or 2D, repeat num_heads times
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    # X_in shape  :  (batch_size, num_key_value, num_hiddens)
    # X_out shape :  (batch_size, num_key_value, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # X_out shape :  (batch_size, num_heads, num_key_value, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # X_out shape :  (batch_size*num_heads, num_key_value, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    # reverse transpose_qkv function
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def test1():
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(
        num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5
    )
    print(attention.eval())
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)


# Suppose input X: (seq_len=2, query_size=3)
# X = [[x11, x12, x13],
#      [x21, x22, x23]]

# Weight matrix W_q: (query_size=3, num_hiddens=4)
# num_heads=2 → head_dim = 2
# Columns 0-1 for head1, columns 2-3 for head2
# W_q =
# [[w11, w12, w13, w14],
#  [w21, w22, w23, w24],
#  [w31, w32, w33, w34]]

# Step 1: Compute the large Q matrix (before splitting heads)
# Q = X @ W_q
# Q shape: (seq_len=2, num_hiddens=4)
# q11 = x11*w11 + x12*w21 + x13*w31  # first row, first column
# q12 = x11*w12 + x12*w22 + x13*w32  # first row, second column
# q13 = x11*w13 + x12*w23 + x13*w33  # first row, third column
# q14 = x11*w14 + x12*w24 + x13*w34  # first row, fourth column

# q21 = x21*w11 + x22*w21 + x23*w31  # second row, first column
# q22 = x21*w12 + x22*w22 + x23*w32  # second row, second column
# q23 = x21*w13 + x22*w23 + x23*w33  # second row, third column
# q24 = x21*w14 + x22*w24 + x23*w34  # second row, fourth column

# Large Q matrix:
# Q = [[q11, q12, q13, q14],
#      [q21, q22, q23, q24]]

# Step 2: Split Q into multiple heads
# head1 takes the first 2 columns
# Q_head1 = Q[:, 0:2]
# Shape: (seq_len=2, head_dim=2)
# Element-wise:
# q11^(head1) = q11 = x11*w11 + x12*w21 + x13*w31
# q12^(head1) = q12 = x11*w12 + x12*w22 + x13*w32
# q21^(head1) = q21 = x21*w11 + x22*w21 + x23*w31
# q22^(head1) = q22 = x21*w12 + x22*w22 + x23*w32

# head2 takes the last 2 columns
# Q_head2 = Q[:, 2:4]
# Shape: (seq_len=2, head_dim=2)
# Element-wise:
# q11^(head2) = q13 = x11*w13 + x12*w23 + x13*w33
# q12^(head2) = q14 = x11*w14 + x12*w24 + x13*w34
# q21^(head2) = q23 = x21*w13 + x22*w23 + x23*w33
# q22^(head2) = q24 = x21*w14 + x22*w24 + x23*w34


