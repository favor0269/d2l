import text_preprocess
import torch
import random


def seq_data_iter_random(corpus, batch_size, num_steps):
    # subseq 1: [a, b, c] num_step = 3
    # a batch: [subseq1, subseq2] batch_size = 2
    # the seq is divided and shuffled: [subseq(i), subseq(j) ...]
    # num_batch = len([subseq(i), subseq(j) ...]) / batch_size

    # start from random offset [0, num_step-1] all closed
    corpus = corpus[random.randint(0, num_steps - 1) :]
    # minor 1 because we need to leave place for label
    num_subseqs = (len(corpus) - 1) // num_steps
    # get initial(start) idx for subseq (length=num_step)
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # shuffle
    random.shuffle(initial_indices)

    def data(pos):
        # pos: initial idx
        return corpus[pos : pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i : i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # each row independently continuous, like:
    # X: tensor([[0, 1, 2, 3, 4], [17, 18, 19, 20, 21]])
    # X: tensor([[5, 6, 7, 8, 9], [22, 23, 24, 25, 26]])
    # RNN and Transformer will need it
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y
        
    
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = text_preprocess.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
        
    def __iter__(self):
        self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
        

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab