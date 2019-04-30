"""This is the file for main model."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from settings import use_cuda, MAX_LENGTH, LAYER_DEPTH, TOCOPY, DROPOUT

class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=LAYER_DEPTH):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers        
    
    def forward(self, input):
        """
        input is a one-hot vector.
        (vocab_size)
        """

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers,
                                      batch_size,
                                      self.hidden_size),
                                      requires_grad=False)

        if use_cuda:
            return result.cuda()
        else:
            return result

class GRU(nn.Module):
    """."""
    def __init__(self, hidden_size, embedding_layer, n_layers=LAYER_DEPTH):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers)

    def forward(self, inputs):
        # embedded is of size (n_batch, seq_len, emb_dim)
        # gru needs (seq_len, n_batch, emb_dim)

        # output (seq_len, batch, hidden_size * num_directions)
        # 1. hidden is the at t = seq_len
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers,
                                      batch_size,
                                      self.hidden_size),
                                      requires_grad=False)

        if use_cuda:
            return result.cuda()
        else:
            return result
