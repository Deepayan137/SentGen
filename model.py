import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import pdb
from gumbel import _gumbel_softmax
class DiscNet(nn.Module):
	def __init__(self, vocab_size, hidden_size, embedding_size, rnn_type, word_dropout):
		super(DiscNet, self).__init__()
		self.hidden_size = hidden_size
		# self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.embedding = nn.Linear(vocab_size, embedding_size)
		self.rnn_type = rnn_type

		if rnn_type == 'rnn':
			rnn = nn.RNN
		elif rnn_type == 'gru':
			rnn = nn.GRU

		self.rnn = rnn(embedding_size, hidden_size, batch_first=True)
		self.rnn2hidden = nn.Linear(hidden_size, hidden_size)
		self.dropout_linear = nn.Dropout(p=word_dropout)
		self.hidden2out = nn.Linear(hidden_size, 1)

	def forward(self, input_, length):
		# input_sequence  batch_size x max_seq_len
		batch_size = input_.size(0)
		sorted_lengths, sorted_idx = torch.sort(length, descending=True)
		input_ = input_[sorted_idx]
		input_embedding = self.embedding(input_)  #batch_size x seq_len

		packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

		_, hidden = self.rnn(packed_input)
		hidden = hidden.squeeze()

		out = self.rnn2hidden(hidden)
		out = nn.functional.tanh(out)
		out = self.dropout_linear(out)
		out = self.hidden2out(out)
		out = nn.functional.sigmoid(out)
		return out

class GenNet(nn.Module):
	def __init__(self, vocab_size, hidden_size, embedding_size, rnn_type, word_dropout=0.2):
		super(GenNet, self).__init__()
		self.hidden_size = hidden_size
		self.rnn_type = rnn_type

		if rnn_type == 'rnn':
			rnn = nn.RNN
		elif rnn_type == 'gru':
			rnn = nn.GRU

		self.rnn = rnn(embedding_size, hidden_size, batch_first=True)
		self.hidden2out = nn.Linear(hidden_size, vocab_size)

	def forward(self, input_):
		out, hidden = self.rnn(input_)
		out = self.hidden2out(out)
		# out = nn.functional.log_softmax(out)
		out = _gumbel_softmax(out, 1.0)
		return out

	