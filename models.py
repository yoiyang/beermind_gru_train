import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as torch_init
import torch.nn.functional as F
from torch.autograd import Variable

class baselineLSTM(nn.Module):
    def __init__(self, cfg):
        super(baselineLSTM, self).__init__()
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.output_dim = cfg['output_dim']
        self.batch_size = cfg['batch_size']
        self.num_layers = cfg['layers']
        self.dropout = cfg['dropout']
        self.bidirect = cfg['bidirectional']
        self.cuda = cfg['cuda']
        self.training = cfg['train']

        # input layer requried?
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            bias=True, dropout=self.dropout, bidirectional=self.bidirect, batch_first = True)

        directions = 2 if self.bidirect else 1

        self.out = nn.Linear(self.hidden_dim * directions, self.output_dim)
        if (self.cuda):
            self.lstm, self.out = self.lstm.cuda(), self.out.cuda()

    def init_hidden(self,batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        directions = 2 if self.bidirect else 1

        return (torch.zeros(self.num_layers * directions,batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * directions,batch_size, self.hidden_dim))


    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        if (self.cuda):
            if ( type(sequence).__module__ == np.__name__):
                sequence = torch.Tensor(sequence).cuda()
            else:
                sequence = sequence.cuda()
            if isinstance(self.hidden, tuple):
                self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())
            else:
                self.hidden = self.hidden.cuda()

        output, self.hidden = self.lstm(sequence, self.hidden)

        del sequence

        output = self.out(output)#.view(sequence_len, -1)

        return output

class baselineGRU(nn.Module):
    def __init__(self, cfg):
        super(baselineGRU, self).__init__()
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.

        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.output_dim = cfg['output_dim']
        self.batch_size = cfg['batch_size']
        self.num_layers = cfg['layers']
        self.dropout = cfg['dropout']
        self.bidirect = cfg['bidirectional']
        self.cuda = cfg['cuda']
        self.training = cfg['train']

        # input layer requried?
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers,
                            bias=True, dropout=self.dropout, bidirectional=self.bidirect, batch_first = True)

        directions = 2 if self.bidirect else 1
        self.out = nn.Linear(self.hidden_dim * directions, self.output_dim)
        if (self.cuda):
            self.gru, self.out = self.gru.cuda(), self.out.cuda()

    def init_hidden(self,batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        directions = 2 if self.bidirect else 1
        return torch.zeros(self.num_layers * directions,batch_size, self.hidden_dim)

    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        if (self.cuda):
            if ( type(sequence).__module__ == np.__name__):
                sequence = torch.Tensor(sequence).cuda()
            else:
                sequence = sequence.cuda()
            if isinstance(self.hidden, tuple):
                self.hidden = (self.hidden[0].cuda(), self.hidden[1].cuda())
            else:
                self.hidden = self.hidden.cuda()

        output, self.hidden = self.gru(sequence, self.hidden)

        del sequence

        output = self.out(output)#.view(sequence_len, -1)

        return output
