"""
PolicyNetwork(nn.Module) and ValueNetwork(nn.Module) for VPG
Same architecture as REINFORCE, adapted for VPG usage
"""
import torch
import torch.nn as nn

# Remove global device check, handled in Agent

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        """
        CNN for processing 96*96*3 RGB images (CarRacing-v3)
        Convolutional layer 1: 32 filters, 8*8 kernel, stride 4, ReLU
        Convolutional layer 2: 64 filters, 4*4 kernel, stride 2, ReLU
        Convolutional layer 3: 64 filters, 3*3 kernel, stride 1, ReLU
        """
        # CNN for processing 96*96*3 RGB images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Calculate CNN output size: 96 -> 23 -> 10 -> 8 = 8x8x64 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        
        # Separate heads for different action types (steering, gas, brake)
        """
        The Car Racing action space is asymmetric:
        Steering: [-1, 1] (tanh works perfectly)
        Gas: [0, 1] (sigmoid)
        Brake: [0, 1] (sigmoid)
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
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Compute means with appropriate activations
        # Steering: [-1, 1] using tanh
        steering_mean = torch.tanh(self.steering_mean(x))
        steering_log_std = self.steering_log_std(x)
        steering_log_std = torch.clamp(steering_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Gas: [0, 1] using sigmoid
        gas_mean = torch.sigmoid(self.gas_mean(x))  
        gas_log_std = self.gas_log_std(x)
        gas_log_std = torch.clamp(gas_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Brake: [0, 1] using sigmoid
        brake_mean = torch.sigmoid(self.brake_mean(x))  
        brake_log_std = self.brake_log_std(x)
        brake_log_std = torch.clamp(brake_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Concatenate means and log stds
        means = torch.cat([steering_mean, gas_mean, brake_mean], dim=1)
        log_stds = torch.cat([steering_log_std, gas_log_std, brake_log_std], dim=1)
        return means, log_stds

    def step(self, state):
        """
        Returns action, log_prob for training
        """
        # State is already a tensor on device from agent.preprocess_state
        means, log_stds = self.forward(state)
        stds = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, stds)
        
        action = dist.sample()
        
        # Calculate log_prob BEFORE clamping to keep math consistent with distribution
        # Sum log probs across dimensions (steering + gas + brake)
        log_prob = dist.log_prob(action).sum(dim=1)
        
        # Now clamp actions for the environment
        action[:, 0] = torch.clamp(action[:, 0], -1, 1)
        action[:, 1] = torch.clamp(action[:, 1], 0, 1)
        action[:, 2] = torch.clamp(action[:, 2], 0, 1)

        return action, log_prob

    def act(self, state):
        """
        Act method: returns only action (for evaluation/testing)
        """
        with torch.no_grad():
            # We can use step but ignore log_prob
            action, _ = self.step(state)
            return action.cpu().numpy()

    def get_log_prob(self, states, actions):
        """
        Compute log probability of actions given states
        Used during policy update
        """
        means, log_stds = self.forward(states)
        stds = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, stds)
        # Important: sum log_probs across the action dimensions
        log_probs = dist.log_prob(actions).sum(dim=1)
        return log_probs


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
        return self.fc3(x).squeeze(-1)  # Flatten to (batch,)
