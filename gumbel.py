from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def _gumbel_softmax_sample(logits, temperature):
    sample = Variable(sample_gumbel(logits.size()))
    if logits.is_cuda:
        sample = sample.cuda()
    y = logits + sample
    return F.softmax(y / temperature)

def _gumbel_softmax(logits, temperature):
    y = _gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros(y.size()).view(-1, shape[-1])
    y_hard.scatter_(1, ind.data.view(-1, 1).cpu(), 1)
    y_hard = y_hard.view(*shape)
    return (Variable(y_hard.cuda()) - y).detach() + y

if __name__ == '__main__':
    import math
    print(_gumbel_softmax(Variable(torch.cuda.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 20000)), 0.8).sum(dim=0))