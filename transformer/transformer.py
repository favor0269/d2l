import os
import sys
import math
import torch
from torch import nn
import multihead_attention
import positional_encoding

# allow importing modules from parent folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modern_rnn import seq2seq
from modern_rnn import translation_and_dataset
from mytools import mytools


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """Add & Norm"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):

    # Muti-Head Attention
    # + Add & Norm
    # + Feed Forward
    # + Add & Norm

    def __init__(
        self,
        query_size,
        key_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        bias=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.attention = multihead_attention.MultiHeadAttention(
            query_size=query_size,
            key_size=key_size,
            value_size=value_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Encoder(nn.Module):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class TransformerEncoder(Encoder):
    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs,
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positional_encoding.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
        **kwargs,
    ):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = multihead_attention.MultiHeadAttention(
            query_size=query_size,
            key_size=key_size,
            value_size=value_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = multihead_attention.MultiHeadAttention(
            query_size=query_size,
            key_size=key_size,
            value_size=value_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(
                batch_size, 1
            )
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class AttentionDecoder(Decoder):
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class TransformerDecoder(AttentionDecoder):
    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        **kwargs,
    ):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = positional_encoding.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
                ),
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # decoder weight
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # encoder-decoder weight
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
        # self._attention_weights
        # │
        # ├── list_A → [None, None, None]
        # │
        # └── list_B → [None, None, None]

    def attention_weights(self):
        return self._attention_weights


def test1():
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    print(encoder_blk(X, valid_lens).shape)

    encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

    decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    X = torch.ones((2, 100, 24))
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print(decoder_blk(X, state)[0].shape)


def test():
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, mytools.decide_gpu_or_cpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = translation_and_dataset.load_data_nmt(
        batch_size, num_steps
    )

    encoder = TransformerEncoder(
        len(src_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    decoder = TransformerDecoder(
        len(tgt_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    net = EncoderDecoder(encoder, decoder)
    train_losses, train_numbers, _ = seq2seq.train_seq2seq(
        net, train_iter, lr, num_epochs, tgt_vocab, device
    )
    plot_data = [loss / num for loss, num in zip(train_losses, train_numbers)]
    mytools.plot_lines(plot_data)

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = seq2seq.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True
        )
        print(
            f"{eng} => {translation}, ",
            f"bleu {seq2seq.bleu(translation, fra, k=2):.3f}",
        )
    enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape(
        (num_layers, num_heads, -1, num_steps)
    )
    print(enc_attention_weights.shape)
    mytools.plot_heatmaps_grid(
        enc_attention_weights,
        max_images=8,
        cols=4,
        xlabel="Key pos",
        ylabel="Query pos",
        title_prefix="Enc head",
    )
