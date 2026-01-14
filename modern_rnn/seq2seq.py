import collections
import math
import torch
from torch import nn
import encoder_and_decoder
from d2l import torch as d2l
import translation_and_dataset

import os
import sys
from chap_rnn import RNN

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mytools import mytools


class Seq2SeqEncoder(encoder_and_decoder.Encoder):
    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state


class Seq2SeqDecoder(encoder_and_decoder.Decoder):

    def __init__(
        self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs
    ):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs is (output, state); we pass encoder state to decoder
        return enc_outputs[1]  # (num_layers, batch_size, num_hiddens)

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)  # -> (num_steps, batch_size, embed_size)

        context = state[-1].repeat(
            X.shape[0], 1, 1
        )  # (num_steps, batch_size, num_hiddens)
        X_and_context = torch.cat(
            (X, context), 2
        )  # (num_steps, batch_size, embed_size + num_hiddens)

        # Return two values:
        # 1. the last layer at each time step (batch_size, num_steps, vocab_size)
        # 2. the final time step at each layer (num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output)
        output = output.permute(1, 0, 2)  # -> (batch_size, num_steps, vocab_size)
        return output, state


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


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # softmax CrossEntropyLoss with mask
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"  # return loss like (batch_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )  # CrossEntropyLoss requires (N, C, ...)
        # (batch_size, num_steps, vocab_size)->(batch_size, vocab_size, num_steps)
        # unweight_loss (batch_size, num_steps)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        # weighted_loss (batch_size,)
        return weighted_loss


def test():
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print(output.shape)

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()
    state = decoder.init_state(encoder(X))
    output, state = decoder(X, state)
    print(output.shape)
    print(state.shape)

    X = torch.ones(2, 3, 4)
    print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    loss = MaskedSoftmaxCELoss()
    print(
        loss(
            torch.ones(3, 4, 10),
            torch.ones((3, 4), dtype=torch.long),
            torch.tensor([4, 2, 0]),
        )
    )


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):

    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.GRU):
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # net includes encoder and decoder
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    train_losses = []
    train_numbers = []
    train_times = []
    for epoch in range(num_epochs):
        timer = mytools.Timer()

        train_loss = 0.0
        train_number = 0.0
        for batch in data_iter:
            optimizer.zero_grad()
            # shape:
            # X             : (batch_size, src_num_steps)
            # X_valid_len   : (batch_size,)
            # Y             : (batch_size, tgt_num_steps)
            # Y_valid_len   : (batch_size,)
            X, X_valid_len, Y, Y_valid_len = [data.to(device) for data in batch]
            bos = torch.tensor(
                [tgt_vocab["<bos>"]] * Y.shape[0], device=device
            ).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # Y_hat: (batch_size, tgt_num_steps, vocab_size)
            # Y: (batch_size, tgt_num_steps)
            l = loss(Y_hat, Y, Y_valid_len)  # (batch_size,)
            l.sum().backward()
            RNN.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                train_loss += l.sum().item()
                train_number += num_tokens.item()

        train_time = timer.stop()
        if (epoch + 1) % 10 == 0:
            print(
                f"epoch: {epoch:3d}, loss: {train_loss/train_number:3.4f}, tokens/time: {train_number/train_time:3.2f}/s"
            )
        train_losses.append(train_loss)
        train_numbers.append(train_number)
        train_times.append(train_time)

    return train_losses, train_numbers, train_times


def predict_seq2seq(
    net,
    src_sentence,
    src_vocab,
    tgt_vocab,
    num_steps,
    device,
    save_attention_weights=False,
):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = translation_and_dataset.truncate_pad(
        src_tokens, num_steps, src_vocab["<pad>"]
    )
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )  # add batch dimension for net
    # pay attention
    # this output is (output, state)
    # we use enc_outputs[1] (state)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0
    )  # (1, 1) [[bos]]
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        # remove batch dimension
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)

    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def main():
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, mytools.decide_gpu_or_cpu()
    train_iter, src_vocab, tgt_vocab = translation_and_dataset.load_data_nmt(
        batch_size, num_steps
    )
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    net = encoder_and_decoder.EncoderDecoder(encoder, decoder)
    train_losses, train_numbers, _ = train_seq2seq(
        net, train_iter, lr, num_epochs, tgt_vocab, device
    )
    plot_data = [loss / num for loss, num in zip(train_losses, train_numbers)]
    mytools.plot_lines(plot_data)

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device
        )
        print(f"{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}")


def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(" "), label_seq.split(" ")
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        # record number of n-gram
        # vocabulary includes n-gram: times (key-value)
        num_matches, label_subs = 0, collections.defaultdict(int)

        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i : i + n])] += 1

        for i in range(len_pred - n + 1):
            if label_subs[" ".join(pred_tokens[i : i + n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i : i + n])] -= 1

        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))

    return score


if __name__ == "__main__":
    main()


# =============================================================
# Seq2Seq (GRU) data flow quick map
#
# Data loader -> (X, X_valid_len, Y, Y_valid_len)
#   X: src ids, shape (batch, src_steps)
#   Y: tgt ids, shape (batch, tgt_steps)  [labels]
#
# Encoder forward
#   X --embedding--> (batch, src_steps, embed)
#       --permute--> (src_steps, batch, embed)
#       --GRU--> encoder outputs, state (num_layers, batch, hidden)
#
# Decoder input prep
#   bos = <bos> ids (batch, 1)
#   dec_input = concat(bos, Y[:, :-1])  # teacher forcing inputs
#
# Decoder forward
#   dec_input --embedding--> (batch, tgt_steps, embed)
#              --permute--> (tgt_steps, batch, embed)
#   context = last encoder state (batch, hidden) repeated over steps
#   cat([embed, context]) -> (tgt_steps, batch, embed+hidden)
#   --GRU--> hidden seq (tgt_steps, batch, hidden)
#   --Linear--> logits (tgt_steps, batch, vocab)
#   --permute--> Y_hat (batch, tgt_steps, vocab)
#
# Loss (MaskedSoftmaxCELoss)
#   pred = Y_hat -> permute to (batch, vocab, tgt_steps) for CrossEntropy
#   label = Y (batch, tgt_steps) integer ids
#   valid_len -> mask padding positions
#   cross entropy per token -> mask -> mean over steps -> (batch,)
#
# Optim
#   l.sum().backward() -> grad clip -> Adam step
# =============================================================
