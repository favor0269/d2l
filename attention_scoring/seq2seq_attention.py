import torch
from torch import nn

import os
import sys

# Make project root importable when running as a script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import attention_scoring
from modern_rnn import encoder_and_decoder
from modern_rnn import seq2seq
from mytools import mytools
from modern_rnn import translation_and_dataset


class AttentionDecoder(encoder_and_decoder.Decoder):

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = attention_scoring.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout
        )

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs (num_steps, batch_size, num_hiddens)
        # hidden_state (num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        # outputs (batch_size, num_steps, num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):

        # 1. query: the last hidden_state (include hidden_state from encoder and dec_input(read before))
        # 2. key and value: encoder's hidden_states (keep constant)
        # 3. we train W_h, W_q, w_v
        # 4. then we get context and cat it to x(dec_input) (this time_step)
        # 5. then we predict next output and get hidden_state for next use

        # enc_outputs (batch_size, num_steps, num_hiddens) remain constant
        # hidden_state (num_layers, batch_size, num_hiddens) renew each x loop
        enc_outputs, hidden_state, enc_valid_lens = state

        X = self.embedding(X).permute(1, 0, 2)  # (num_steps,batch_size,embed_size)
        outputs, self._attention_weights = [], []

        for x in X:
            # x: (batch_size, embed_size)
            query = torch.unsqueeze(
                hidden_state[-1], dim=1
            )  # query (batch_size, 1, num_hiddens)
            # here keys = values = enc_outputs
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens
            )  # remove <pad>
            # context (batch_size, 1, num_hiddens)

            # 1. unsqueeze x -> (batch_size, 1, embed_size)
            # 2. cat -> (batch_size, 1, num_hiddens + embed_size)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)

            # x.permute -> (1, batch_size, embed_size + num_hiddens)
            # out (1, batch_size, num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)

            # record weight every time
            self._attention_weights.append(self.attention.attention_weights)

        # outputs shape
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))

        # finally return (batch_size, num_steps, vocab_size)
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


def test():
    encoder = seq2seq.Seq2SeqEncoder(
        vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2
    )
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(
        vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2
    )
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print(
        "output",
        output.shape,
        "state_len",
        len(state),
        "enc_outputs_shape",
        state[0].shape,
        "hidden_layers",
        len(state[1]),
        "hidden_shape",
        state[1][0].shape,
    )


def main():
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, mytools.decide_gpu_or_cpu()
    train_iter, src_vocab, tgt_vocab = translation_and_dataset.load_data_nmt(
        batch_size, num_steps
    )

    encoder = seq2seq.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    net = encoder_and_decoder.EncoderDecoder(encoder, decoder)

    train_losses, train_numbers, _ = seq2seq.train_seq2seq(
        net, train_iter, lr, num_epochs, tgt_vocab, device
    )
    plot_data = [loss / num for loss, num in zip(train_losses, train_numbers)]
    mytools.plot_lines(plot_data)
    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = seq2seq.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True
        )
        print(f"{eng} => {translation}, bleu {seq2seq.bleu(translation, fra, k=2):.3f}")

main()