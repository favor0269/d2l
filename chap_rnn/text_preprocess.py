import os
import re
import collections

# abspath return the absolute path
# dirname return the directory of the file (remove the file name)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# join the path together
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "timemachine")


def read_time_machine():
    data_dir = DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR not found: {data_dir}. Please place the banana-detection dataset there."
        )
    txt_path = os.path.join(data_dir, "timemachine.txt")
    with open(txt_path, "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("Error, unknown token: " + token)


class Vocabulary:
    
    # features:
    # len(vocab)
    # vocab['apple'] vocab[ ['token1', 'token2'] ]
    # vocab.to_tokens(idx) vocab.to_tokens([idx1, idx2 ...]) 
    # an attribute begins with _ meaning that private 
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        # _token_freq: [('apple', 3), ('banana', 2), ('orange', 1)] if min_freq = 2
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # <unk> is a special key -- unknown
        self._idx_to_token = ["<unk>"] + reserved_tokens
        # enumerate return a iter: [(0, '<unk>'), (1, '<pad>'), (2, 'apple'), (3, 'banana')]
        self._token_to_idx = {token: idx for idx, token in enumerate(self._idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self._token_to_idx:  # search base on key, O(1)
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self._idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, self.unk)  # get(token, default)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self._idx_to_token[indices]
        return [self._idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocabulary(tokens)
    
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
