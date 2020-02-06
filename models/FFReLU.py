""" Fully-connected, feedforward ReLU networks """

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_classes = config['n_classes']
        self.input_shape = config['input_shape']
        self.n_hidden = config['n_hidden_units']
        self.n_layers = config['n_hidden_layers']

        self._make_layers()

    def _make_layers(self):
        self.input_layer = nn.Linear(in_features = self.input_shape[1],
                            out_features = self.n_hidden)
        self.hidden_layers = []
        for i in range(self.n_layers):
          self.hidden_layers.append(nn.Linear(in_features = self.n_hidden,
                                              out_features = self.n_hidden))
        self.hidden = nn.Sequential(*self.hidden_layers)
        self.output_layer = nn.Linear(in_features = self.n_hidden,
                            out_features = self.n_classes)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for L in self.hidden_layers:
          x = F.relu(L(x))
        x = self.output_layer(x)

        return x
