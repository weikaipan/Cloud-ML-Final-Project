"""This is the file for main model."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from configs import use_cuda, LAYER_DEPTH, DROPOUT, EMBEDDING_SIZE, OUTPUT_DIM

class BaseLine(nn.Module):
    # This is our baseline model.
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 embedding_dim=EMBEDDING_SIZE,
                 n_layers=LAYER_DEPTH,
                 dropout=DROPOUT,
                 output_dim=OUTPUT_DIM,
                 topology='RNN'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers     
        
        # Embedding layer.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Recurrent Layer.
        self.rnn = nn.RNN(embedding_dim, hidden_size)
        
        # Linear layer.
        self.fc = nn.Linear(hidden_size, output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequence):
        """
        input is a one-hot vector.
        (seq_len, batch_size)
        """
        # seq_len, batch_size, embedding_dim
        embeds = self.embedding(sequence)
        # seq_len, batch, input_size
        
        output, hidden = self.rnn(embeds)
        # output: seq_len, batch, num_directions * hidden_size
        # hidden: num_layers * num_directions, batch, hidden_size (last layer)
        out = self.fc(hidden.squeeze(0))
        return out

class RNN(nn.Module):
    # This is our baseline model.
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 embedding_dim=EMBEDDING_SIZE,
                 n_layers=LAYER_DEPTH,
                 dropout=DROPOUT,
                 output_dim=OUTPUT_DIM,
                 topology='RNN'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers     
        
        # Embedding layer.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Recurrent Layer.
        self.topology = topology        
        if self.topology == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_size)
        elif self.topology == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size)
        
        # Linear layer.
        self.fc = nn.Linear(hidden_size, output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequence, text_len):
        """
        input is a one-hot vector.
        (seq_len, batch_size)
        """
        # sequence: seq_len, batch_size, embedding_dim
        embeds = self.embedding(sequence)
        print("embedding shape")
        print(embeds.shape)
        # 
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, text_len)
        print("packed embedding shape")
        print(packed_embeds.shape)
        # seq_len, batch, input_size
        output, hidden = self.rnn(packed_embeds)
        
        # output: seq_len, batch, num_directions * hidden_size
        # hidden: num_layers * num_directions, batch, hidden_size (last layer)
        out = self.fc(hidden.squeeze(0))
        return out
