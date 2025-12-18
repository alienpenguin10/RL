import torch
import torch.nn as nn

class NNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.output(x)
        
        return output
