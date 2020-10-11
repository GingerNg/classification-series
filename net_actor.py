import torch.nn as nn
import torch.nn.functional as F
from os.path import dirname
from os import getcwd
from os.path import realpath
from sys import argv


class NetActor(nn.Module):

    def __init__(self, args, state_vector_size, action_vector_size, hidden_layer_size_list):
        super(NetActor, self).__init__()
        self.args = args

        self.state_vector_size = state_vector_size
        self.action_vector_size = action_vector_size
        self.layer_sizes = hidden_layer_size_list
        self.layer_sizes.append(action_vector_size)

        self.nn_layers = []
        self._create_net()

    def _create_net(self):
        prev_layer_size = self.state_vector_size
        for next_layer_size in self.layer_sizes:
            next_layer = nn.Linear(prev_layer_size, next_layer_size)
            prev_layer_size = next_layer_size
            self.nn_layers.append(next_layer)

    def forward(self, torch_state):
        activations = torch_state
        for i, layer in enumerate(self.nn_layers):
            if i != len(self.nn_layers)-1:
                activations = F.relu(layer(activations))
            else:
                activations = layer(activations)

        probs = F.softmax(activations, dim=-1)
        return probs


actor_nn = NetActor(self.args, 4, 2, [128])
actor_optimizer = optim.Adam(self.actor_nn.parameters(), lr=args.learning_rate)