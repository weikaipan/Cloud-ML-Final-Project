"""This is the file for main model."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from configs import use_cuda, LAYER_DEPTH, DROPOUT, EMBEDDING_SIZE, OUTPUT_DIM

class RNN(nn.Module):
    # This is our baseline model.
    def __init__(self, vocab_size, hidden_size, embedding_dim=EMBEDDING_SIZE, n_layers=LAYER_DEPTH, output_dim=OUTPUT_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers     
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, sequence):
        """
        input is a one-hot vector.
        (seq_len, batch_size)
        """
        try:
            embeds = self.embedding(sequence)
        except Exception as e:
            print(sequence)
            print(e)
            quit()
        # seq_len, batch_size, embedding_dim
        # seq_len, batch, input_size
        
        output, hidden = self.rnn(embeds)
        # output: seq_len, batch, num_directions * hidden_size
        # hidden: num_layers * num_directions, batch, hidden_size (last layer)
        out = self.fc(hidden.squeeze(0))
        print("out shape {}".format(out.shape))
        return out

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
        # input is of size (seq_len, n_batch)

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
