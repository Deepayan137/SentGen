import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model import DiscNet, GenNet
from utils import noise, real_data_target, fake_data_target, to_var
from loader import PTB
from utils import base_opts, format_fake, get_sentences, one_hot
from argparse import ArgumentParser
import pdb
from collections import defaultdict, Counter, OrderedDict
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
def train_discriminator(optimizer, real_data, length, fake_data):
    optimizer.zero_grad()
    prediction_real = discriminator(real_data, length)
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()
    prediction_fake = discriminator(fake_data, length)
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    optimizer.step()
    return error_real.data[0] + error_fake.data[0], prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
	optimizer.zero_grad()
	prediction = discriminator(fake_data, length)
	error = loss(prediction, real_data_target(prediction.size(0)))
	error.backward()
	optimizer.step()
	return error.data[0]


if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	splits = ['train', 'valid'] 
	datasets = OrderedDict()
	for split in splits:
		datasets[split] = PTB(
		data_dir=args.data_dir,
		split=split,
		create_data=args.create_data,
		max_sequence_length=args.max_sequence_length,
		min_occ=args.min_occ
		)
	word2index, index2word = datasets['train'].w2i, datasets['train'].i2w
	discriminator = DiscNet(vocab_size=datasets['train'].vocab_size,
							hidden_size = args.hidden_size,
        					embedding_size=args.embedding_size,
        					rnn_type=args.rnn_type,
        					word_dropout=args.word_dropout)
	generator = GenNet(vocab_size=datasets['train'].vocab_size,
							hidden_size = args.hidden_size,
        					embedding_size=args.embedding_size,
        					rnn_type=args.rnn_type,
        					word_dropout=args.word_dropout)

	if torch.cuda.is_available():
	    discriminator = discriminator.cuda()
	    generator = generator.cuda()
	# Optimizers
	d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
	g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))

	# Loss function
	loss = nn.BCELoss()
	d_steps = 1 
	num_epochs = 200
	num_test_samples = 2
	test_noise = noise(num_test_samples)
	for epoch in range(num_epochs):
		for split in splits:
			data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                pin_memory=torch.cuda.is_available())

			if split == 'train':
				discriminator.train()
				generator.train()
			else:
				discriminator.eval()
				generator.eval()
			for iteration, batch in enumerate(data_loader):
				batch_size = batch['input'].size(0)
				for k, v in batch.items():
					if torch.is_tensor(v):
						batch[k] = to_var(v)

				real_data = to_var(one_hot(batch['input']))
				length = batch['length']
				if torch.cuda.is_available(): 
					real_data = real_data.cuda()
				# Generate fake data
				noise_ = noise(real_data.size(0))
				fake_data = generator(noise_).detach()
				d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
																	real_data, length, fake_data)
				fake_data = generator(noise_)
				g_error = train_generator(g_optimizer, fake_data)

				if iteration%100 == 0:
					print(get_sentences(generator(test_noise), index2word))
					print('{} d-Loss: {:.4f} g-Loss: {:.4f}'.format(
						epoch, d_error, g_error))
			if split == 'train':
				checkpoint_path = os.path.join('bin', "E%i.pytorch"%(epoch))
				torch.save(generator.state_dict(), checkpoint_path)
				print("Model saved at %s"%checkpoint_path)

