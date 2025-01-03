# code adapted from https://github.com/wendelinboehmer/dcg
# Outputs filtered to only correspond to the top M assignments.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FilteredAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FilteredAgent, self).__init__()
        self.args = args
        self.M = args.env_args["M"]

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, self.M+1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h
