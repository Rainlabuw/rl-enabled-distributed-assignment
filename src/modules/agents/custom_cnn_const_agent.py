# code adapted from https://github.com/wendelinboehmer/dcg

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CustomCNNConstellationAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CustomCNNConstellationAgent, self).__init__()
        self.args = args

        self.M = args.env_args["M"]
        self.L = args.env_args["L"]
        self.N = args.env_args["N"]
        other_info_size = input_shape - self.M*self.L - self.N*self.M*self.L - self.N*(self.M//2)*self.L

        num_filters = args.num_filters
        hidden_dim = args.hidden_dim

        # Agent-wise convolution: convolve each agent with all of its tasks
        self.local_agent_conv = nn.Conv2d(in_channels=self.L, out_channels=num_filters, kernel_size=(1, self.M))
        self.neigh_agent_conv = nn.Conv2d(in_channels=self.L, out_channels=num_filters, kernel_size=(1, self.M))
        self.global_agent_conv = nn.Conv2d(in_channels=self.L, out_channels=num_filters, kernel_size=(1, self.M//2))

        # Task-wise convolution: convolve each task with all of its agents
        self.neigh_task_conv = nn.Conv2d(in_channels=self.L, out_channels=num_filters, kernel_size=(self.N, 1))
        self.global_task_conv = nn.Conv2d(in_channels=self.L, out_channels=num_filters, kernel_size=(self.N, 1))

        # Feature bringing in other information
        self.other_info_fc = nn.Linear(other_info_size, num_filters)

        num_features_combined = num_filters + num_filters*(self.N)*2 + \
                                num_filters*self.M + num_filters*(self.M//2) + num_filters

        # Define the MLP layers
        self.fc1 = nn.Linear(num_features_combined, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.M + 1)  # Output layer is Q-Values for each action and the no-op action

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_() #not used here

    def forward(self, obs, hidden_state):
        # Retrieve the local, neighboring, and neighboring other benefits from obs
        local_benefits_end = self.M*self.L
        neighboring_benefits_end = local_benefits_end + self.N*self.M*self.L
        global_benefits_end = neighboring_benefits_end + self.N*(self.M//2)*self.L
        local_benefits = obs[:, :local_benefits_end].reshape(-1, 1, self.M, self.L)
        neighboring_benefits = obs[:, local_benefits_end:neighboring_benefits_end].reshape(-1, self.N, self.M, self.L)
        global_benefits = obs[:, neighboring_benefits_end:global_benefits_end].reshape(-1, self.N, self.M//2, self.L)
        other_info = obs[:, global_benefits_end:]

        # all benefits initially have shapes of [batch_size, (N,) M, L]
        local_benefits = local_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, 1, M]
        neighboring_benefits = neighboring_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]
        global_benefits = global_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]

        # Agent-wise convolution
        local_agent_features = F.relu(self.local_agent_conv(local_benefits))
        local_agent_features = local_agent_features.reshape(local_agent_features.size(0), -1)  # Flatten the features
        neighboring_agent_features = F.relu(self.neigh_agent_conv(neighboring_benefits))
        neighboring_agent_features = neighboring_agent_features.reshape(neighboring_agent_features.size(0), -1)  # Flatten the features
        global_agent_features = F.relu(self.global_agent_conv(global_benefits))
        global_agent_features = global_agent_features.reshape(global_agent_features.size(0), -1)

        # Task-wise convolution
        neighboring_task_features = F.relu(self.neigh_task_conv(neighboring_benefits))
        neighboring_task_features = neighboring_task_features.reshape(neighboring_task_features.size(0), -1)
        global_task_features = F.relu(self.global_task_conv(global_benefits))
        global_task_features = global_task_features.reshape(global_task_features.size(0), -1)

        # Other information
        other_info_features = F.relu(self.other_info_fc(other_info))
        
        # Combine and flatten features from row and column convolutions
        combined_features = torch.cat((local_agent_features, neighboring_agent_features, global_agent_features,
                                       neighboring_task_features, global_task_features, other_info_features), dim=1)
        
        # MLP processing
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        
        return x, None #no hidden state