import torch
from torch import nn
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # num_hiddens is embedding d
        # dropout to avoid model depend on position excessively
        # (max_len, num_hiddens)
        pe = torch.zeros(max_len, num_hiddens)

        # (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # (num_hiddens // 2,)
        # 10000^(2j/d)
        div_term = torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )

        # even: sin
        pe[:, 0::2] = torch.sin(position / div_term)
        # example:
        # position = [[0],
        #     [1],
        #     [2],
        #     [3]]   # shape (4,1)
        # div_term = [10, 100, 1000]   # shape (3,)

        # position / div_term
        # = [[0/10,   0/100,   0/1000],
        # [1/10,   1/100,   1/1000],
        # [2/10,   2/100,   2/1000],
        # [3/10,   3/100,   3/1000]]

        # odd: cos
        pe[:, 1::2] = torch.cos(position / div_term)

        # add batch size
        # (1, max_len, num_hiddens)

        self.P = pe.unsqueeze(0)
        
    def forward(self, X):
        """
        X: (batch_size, seq_len, num_hiddens)
        """
        X = X + self.P[:, : X.size(1), :].to(X.device)
        return self.dropout(X)


def test1():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    P = P[0]
    print(P.shape)
    plot_positional_encoding(P)


def plot_positional_encoding(P):
    """Plot heatmap for positional encodings.

    P shape: (positions, hidden_dim)
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(P, cmap="viridis", aspect="auto")
    plt.xlabel("Hidden dimension")
    plt.ylabel("Position")
    plt.colorbar(label="Value")
    plt.title("Positional Encoding Heatmap")
    plt.tight_layout()
    plt.show()
    
    
test1()
# pe shape = (max_len, 8)

# |   0        |   1        |   2           |   3           |   4              |   5              |   6                 |   7                 |
# | sin(pos)   | cos(pos)   | sin(pos/10)   | cos(pos/10)   | sin(pos/100)     | cos(pos/100)     | sin(pos/10000)      | cos(pos/10000)      |
# --------------------------------------------------------------------------------------------------------------------------------

# pos = 0
# |  sin(0)    |  cos(0)    |  sin(0/10)    |  cos(0/10)    |  sin(0/100)      |  cos(0/100)      |  sin(0/10000)       |  cos(0/10000)       |
# |  0.000     |  1.000     |  0.000        |  1.000        |  0.000           |  1.000           |  0.000              |  1.000              |
# --------------------------------------------------------------------------------------------------------------------------------

# pos = 1
# |  sin(1)    |  cos(1)    |  sin(0.1)     |  cos(0.1)     |  sin(0.01)       |  cos(0.01)       |  sin(0.0001)        |  cos(0.0001)        |
# |  0.841     |  0.540     |  0.100        |  0.995        |  0.010           |  0.999           |  0.0001             |  1.000              |
# --------------------------------------------------------------------------------------------------------------------------------

# pos = 5
# |  sin(5)    |  cos(5)    |  sin(0.5)     |  cos(0.5)     |  sin(0.05)       |  cos(0.05)       |  sin(0.0005)        |  cos(0.0005)        |
# | -0.959     |  0.284     |  0.479        |  0.878        |  0.050           |  0.999           |  0.0005             |  1.000              |
# --------------------------------------------------------------------------------------------------------------------------------

# pos = 20
# |  sin(20)   |  cos(20)   |  sin(2.0)     |  cos(2.0)     |  sin(0.20)       |  cos(0.20)       |  sin(0.0020)        |  cos(0.0020)        |
# |  0.913     |  0.408     |  0.909        | -0.416        |  0.199           |  0.980           |  0.0020             |  1.000              |
# --------------------------------------------------------------------------------------------------------------------------------
