import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from collections import defaultdict, Counter, OrderedDict
from torch.autograd import Variable	
import pdb
def noise(size):
	n = Variable(torch.randn(size, 60, 300))
	if torch.cuda.is_available(): return n.cuda() 
	return n


def real_data_target(size):
	data = Variable(torch.ones(size, 1))
	if torch.cuda.is_available(): return data.cuda()
	return data

def fake_data_target(size):
	data = Variable(torch.zeros(size, 1))
	if torch.cuda.is_available(): return data.cuda()
	return data
def format_fake(fake_data):
    f_data = torch.zeros(fake_data.data.size(0), 60).long()
    for i in range(fake_data.size(0)):
        _, indices =  fake_data[i].max(1, keepdim=True)
        f_data[i]=indices.data
    if torch.cuda.is_available():
        f_data = f_data.cuda()
    return to_var(f_data)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
def get_sentences(data, i2w):
    data = format_fake(data)
    def sentence(seq):
        # pdb.set_trace()
        return ' '.join(list(map(lambda x:i2w[str(x)], seq.data.cpu().numpy())))
    return '\n'.join([sentence(seq) for seq in data])
class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def one_hot(data):
    batch_size, seq_len = data.size(0), data.size(1)
    one_h = torch.zeros(batch_size, seq_len, 9877)
    for i in range(one_h.size(0)):
        for j in range(one_h.size(1)):
            try:
                loc = data[i, j].data.cpu().numpy()[0]
                one_h[i, j, loc] = 1
            except IndexError:
                pdb.set_trace()
    return one_h
def base_opts(parser):
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)