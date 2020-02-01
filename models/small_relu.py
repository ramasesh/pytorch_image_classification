import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.input_shape = config['input_shape'] 
        self.n_hidden = config['n_hidden']
        self.n_classes = config['n_classes'] 

        self.L1 = nn.Linear(in_features = self.input_shape[1],
                            out_features = self.n_hidden)
        self.L2 = nn.Linear(in_features = self.n_hidden,
                            out_features = self.n_hidden)
        self.L3 = nn.Linear(in_features = self.n_hidden,
                            out_features = self.n_classes)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.L3(x)

        return x
