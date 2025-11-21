"""
PolicyNetwork(nn.Module) and ValueNetwork(nn.Module)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Can use nn.Sequential for this - gives plain, feed-forward networks without custom logic between layers
        """
        Convolutional layer 1: 32 filters, 8*8 kernel, stride 4, ReLU
        Convolutional layer 2: 64 filters, 4*4 kernel, stride 2, ReLU
        Convolutional layer 3: 64 filters, 3*3 kernel, stride 1, ReLU
        Flatten layer
        Fully connected: 512 neurons, ReLU
        """
        # CNN for processing 96*96*3 RGB images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # self.flatten = nn.Flatten() # only needed if using nn.Sequential

        # Calculate CNN output size: 96 -> 23 -> 10 -> 8 = 8x8x64 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        
        # We will use seperate heads for different action types (steering, gas, brake)
        """
        The Car Racing action space is asymmetric:​
        Steering: [-1, 1] (tanh works perfectly)
        Gas: [0, 1] (tanh outputs negative values - using sigmoid!)
        Brake: [0, 1] (tanh outputs negative values - using sigmoid!)
        """
        self.steering_mean = nn.Linear(64, 1)  
        self.steering_log_std = nn.Linear(64, 1)  

        self.gas_mean = nn.Linear(64, 1)  
        self.gas_log_std = nn.Linear(64, 1) 

        self.brake_mean = nn.Linear(64, 1)  
        self.brake_log_std = nn.Linear(64, 1) 


        # Constants for log_std clamping
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2


    def forward(self, x):
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0

         # CNN processing
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten - reshapes the tensor so the first dimension stays as the batch size and the rest are collapsed into one
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Compute means with appropriate activations
        # Steering: [-1, 1] using tanh
        steering_mean =    torch.tanh( self.steering_mean(x))
        steering_log_std = self.steering_log_std(x)
        steering_log_std = torch.clamp(steering_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Gas: [0, 1] using sigmoid
        gas_mean = torch.sigmoid( self.gas_mean(x))  
        gas_log_std = self.gas_log_std(x)
        gas_log_std = torch.clamp(gas_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Brake: [0, 1] using sigmoid
        brake_mean = torch.sigmoid( self.brake_mean(x))  
        brake_log_std = self.brake_log_std(x)
        brake_log_std = torch.clamp(brake_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Concatanate means and log stds
        means = torch.cat([steering_mean, gas_mean, brake_mean], dim=1)
        log_stds = torch.cat([steering_log_std, gas_log_std, brake_log_std], dim=1)
        return means, log_stds

    def select_action(self, state):
        # Compute policy distribution WITH gradients (needed for log_prob backprop)
        # means = torch.tensor([[0.0, 0.5, 0.5], 3 components (steering, gas, brake) for each state in the batch
        #               .....batch size.....])
        # stds = torch.tensor([[0.1, 0.1, 0.1],
        #              .....batch size.....])
        means, log_stds = self.forward(state)
        stds = torch.exp(log_stds)

        # Create a Gaussian distribution with the mean and std
        dist = torch.distributions.Normal(means, stds)
        
        # Sample action (detach since it's random - we don't want gradients through the sample)
        # action = torch.tensor([[ 0.05,  0.60,  0.48],
        #                     .....batch size.....])
        action = dist.sample().detach() # prevents gradients from flowing through the sampled actions

        # Clamp to valid ranges
        # action is a 2D tensor where the first dimension is the batch and 
        # the second dimension represents action components.
        action[:, 0] = torch.clamp(action[:, 0], -1, 1) # steering
        action[:, 1] = torch.clamp(action[:, 1], 0, 1) # Gas: [0, 1] using sigmoid
        action[:, 2] = torch.clamp(action[:, 2], 0, 1) # Brake: [0, 1] using sigmoid

        # Compute log probability WITH gradients (through means/log_stds, not through action)
        # RECALL: essential for computing policy gradients in REINFORCE - ∇θ J(θ) = E[∇θ log πθ(a|s) * G_t]
        # log_prob will have gradients through policy parameters (means, log_stds) but not through action
        log_prob = dist.log_prob(action)
        return action, log_prob

        
    def evaluate_actions(self, states, actions):
        # Evaluate log probs (and entropy) for given states-actions pairs during training
        with torch.no_grad():
            means, log_stds = self.forward(states)
            stds = torch.exp(log_stds)
            # Create a Gaussian distribution with the mean and std
            dist = torch.distributions.Normal(means, stds)
            # Compute log probability of the taken actions
            log_probs = dist.log_prob(actions).sum(dim=1)
            # Can compute entropy here if needed - maybe later for PPO
            entropy = dist.entropy().sum(dim=1)
            return log_probs #, entropy
    

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Same architecture as PolicyNetwork, but with a single output for state value V(s)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # Could do .squeeze(-1) but not necessary here
