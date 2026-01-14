import os
import sys

# Make project root importable so sibling packages resolve
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

import math
import torch
from torch import nn
from chap_rnn import seq_dataset, text_preprocess
from chap_rnn import RNN_concise
from mytools import mytools


MASK_TOKEN = "<mask>"


def build_mlm_loader(batch_size: int, num_steps: int, use_random_iter: bool = False):
	"""Create a data iterator and vocab with an added <mask> token for MLM."""
	lines = text_preprocess.read_time_machine()
	tokens = text_preprocess.tokenize(lines, token="char")
	vocab = text_preprocess.Vocabulary(tokens, reserved_tokens=[MASK_TOKEN])
	corpus = [vocab[token] for line in tokens for token in line]
	if use_random_iter:
		data_iter_fn = seq_dataset.seq_data_iter_random
	else:
		data_iter_fn = seq_dataset.seq_data_iter_sequential

	def iterator_factory():
		return data_iter_fn(corpus, batch_size, num_steps)

	return iterator_factory, vocab


def mask_inputs(X: torch.Tensor, vocab, mask_prob: float = 0.15):
	"""Apply BERT-style masking to a batch of token ids."""
	mask_idx = vocab[MASK_TOKEN]
	X_masked = X.clone()
	labels = torch.full_like(X, fill_value=-100)  # ignore index
	mask = torch.rand_like(X_masked.float()) < mask_prob
	labels[mask] = X[mask]

	# 80% replace with <mask>, 10% random token, 10% keep
	random_draw = torch.rand_like(X_masked.float())
	mask_replace = mask & (random_draw < 0.8)
	mask_random = mask & (random_draw >= 0.8) & (random_draw < 0.9)
	mask_keep = mask & (random_draw >= 0.9)

	X_masked[mask_replace] = mask_idx
	if mask_random.any():
		random_tokens = torch.randint(low=0, high=len(vocab), size=X_masked.shape, device=X.device)
		X_masked[mask_random] = random_tokens[mask_random]
	# mask_keep leaves the original token

	return X_masked, labels


def train_epoch_mlm(net, train_iter, loss_fn, optimizer, device):
	num_tokens, total_loss = 0, 0.0
	for X, _ in train_iter:
		X = X.to(device)
		X_masked, labels = mask_inputs(X, net.vocab, mask_prob=0.15)
		labels = labels.to(device)
		state = net.begin_state(device, batch_size=X.shape[0])
		y_hat, state = net(X_masked, state)
		labels_flat = labels.T.reshape(-1)
		mask = labels_flat != -100
		if mask.any():
			l = loss_fn(y_hat[mask], labels_flat[mask])
			optimizer.zero_grad()
			l.backward()
			nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
			optimizer.step()
			total_loss += l.item() * mask.sum().item()
			num_tokens += mask.sum().item()

	return math.exp(total_loss / num_tokens) if num_tokens > 0 else float("inf")


def train_mlm(net, iterator_factory, num_epochs, lr, device):
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	ppls = []
	for epoch in range(num_epochs):
		train_iter = iterator_factory()
		ppl = train_epoch_mlm(net, train_iter, loss_fn, optimizer, device)
		ppls.append(ppl)
		if (epoch + 1) % 10 == 0:
			print(f"epoch {epoch+1:4d}, ppl(masked): {ppl:3.4f}")
	return ppls


def main():
	batch_size, num_steps = 32, 35
	iterator_factory, vocab = build_mlm_loader(batch_size, num_steps, use_random_iter=False)
	device = mytools.decide_gpu_or_cpu()

	num_hiddens = 256
	num_layers = 2
	lstm_layer = nn.LSTM(
		input_size=len(vocab),
		hidden_size=num_hiddens,
		num_layers=num_layers,
		bidirectional=True,
	)

	net = RNN_concise.RNNModel(lstm_layer, len(vocab)).to(device)
	# attach vocab so mask_inputs can access mask idx length
	net.vocab = vocab

	num_epochs, lr = 200, 1.0
	ppls = train_mlm(net, iterator_factory, num_epochs, lr, device)
	mytools.plot_lines(ppls)


if __name__ == "__main__":
	main()

