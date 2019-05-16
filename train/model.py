"""This is the file for main model."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# from train.configs import use_cuda, LAYER_DEPTH, DROPOUT, EMBEDDING_SIZE, OUTPUT_DIM, CNN_N_FILTERS
from train.configs import use_cuda, LAYER_DEPTH, DROPOUT, EMBEDDING_SIZE, OUTPUT_DIM, CNN_N_FILTERS
from train.configs import BIDIRECTIONAL, HIDDEN_DIM

class BaseLine(nn.Module):
    # This is our baseline model.
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 embedding_dim=EMBEDDING_SIZE,
                 n_layers=LAYER_DEPTH,
                 dropout=DROPOUT,
                 output_dim=OUTPUT_DIM):
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
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 embedding_dim=EMBEDDING_SIZE,
                 n_layers=LAYER_DEPTH,
                 bidirectional=BIDIRECTIONAL,
                 dropout=DROPOUT,
                 # pad_idx=PAD_IDX,
                 output_dim=OUTPUT_DIM,
                 topology='RNN'):
        # PAD_IDX=TEXT.vocab.stoi[TEXT.pad_token]
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Recurrent Layer.
        self.topology = topology
        if self.topology == 'RNN':
            # self.rnn = nn.RNN(embedding_dim, hidden_size)
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_size,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)

        elif self.topology == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size)

        # Linear layer.
        self.fc = nn.Linear(hidden_size * 2, output_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, text_len):
        """
        sequence is a one-hot vector.
        (seq_len, batch_size)
        """
        # sequence: seq_len, batch_size, embedding_dim
        # embeds = self.embedding(sequence)
        embeds = self.dropout(self.embedding(sequence))
        print("embedding shape")
        print(embeds.shape)
        #
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, text_len)

        # packed sequence object, all rnn modules accept this type
        # output, hidden = self.rnn(packed_embeds)
        packed_output, (hidden, cell) = self.rnn(packed_embeds)

        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output: seq_len, batch, num_directions * hidden_size
        # hidden: num_layers * num_directions, batch, hidden_size (last layer)
        # hidden = self.dropout(hidden)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        out = self.fc(hidden.squeeze(0))
        return out

class CNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 filter_sizes,
                 n_filters=CNN_N_FILTERS,
                 embedding_dim=EMBEDDING_SIZE,
                 n_layers=LAYER_DEPTH,
                 dropout=DROPOUT,
                 output_dim=OUTPUT_DIM,
                 topology='CNN'):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(f, embedding_dim))
                                    for f in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        sequence = sequence.permute(1, 0)
        embeds = self.embedding(sequence)
        embeds = embeds.unsqueeze(1)
        conv_list = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        pool_list = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_list]
        cat = self.dropout(torch.cat(pool_list, dim=1))
        result = self.fc(cat)
        return result
