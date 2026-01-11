"""Train GRU on Li Bai poems using existing RNN_concise/RNN trainer."""

import sys
from pathlib import Path

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from chap_rnn import text_preprocess  # noqa: E402
from chap_rnn import seq_dataset, RNN_concise, RNN  # noqa: E402
from mytools import mytools  # noqa: E402


def load_lines_from_txt(path: Path):
	lines = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				lines.append(line)
	if not lines:
		raise ValueError(f"No non-empty lines in {path}")
	return lines


def build_char_vocab(lines, max_tokens=-1):
	tokens = [list(line) for line in lines]  # char-level (keeps Chinese)
	vocab = text_preprocess.Vocabulary(tokens)
	corpus = [vocab[token] for line in tokens for token in line]
	if max_tokens > 0:
		corpus = corpus[:max_tokens]
	return corpus, vocab


class SimpleSeqLoader:
	"""Wrap corpus into seq_dataset iterator for RNN.train_all."""

	def __init__(self, corpus, batch_size, num_steps, use_random_iter=False):
		self.corpus = corpus
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.data_iter_fn = (
			seq_dataset.seq_data_iter_random if use_random_iter else seq_dataset.seq_data_iter_sequential
		)

	def __iter__(self):
		return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def main():
	txt_path = ROOT_DIR / "data" / "Li_Bai" / "Tang-Li_Bai.txt"
	if not txt_path.exists():
		raise FileNotFoundError(f"poems.txt not found at {txt_path}")

	lines = load_lines_from_txt(txt_path)
	corpus, vocab = build_char_vocab(lines)

	batch_size, num_steps = 32, 35
	train_iter = SimpleSeqLoader(corpus, batch_size, num_steps, use_random_iter=False)

	device = mytools.decide_gpu_or_cpu()
	num_hiddens = 256
	gru_layer = nn.GRU(len(vocab), num_hiddens)
	net = RNN_concise.RNNModel(gru_layer, len(vocab)).to(device)

	num_epochs, lr = 200, 1
	total_perplexities = RNN.train_all(
		net,
		train_iter,
		vocab,
		lr,
		num_epochs,
		device,
		use_random_iter=False,
		predict_prefix_text="君不见",
		num_preds=80,
	)
	mytools.plot_lines(total_perplexities, xlabel="epoch", ylabel="perplexity", title="Li Bai GRU")


if __name__ == "__main__":
	main()
