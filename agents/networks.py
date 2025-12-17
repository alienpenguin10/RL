"""
PolicyNetwork(nn.Module) and ValueNetwork(nn.Module) for VPG
Same architecture as REINFORCE, adapted for VPG usage
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F


# Remove global device check, handled in Agent

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet()
        # ConvNet output with (4, 84, 84) input: 256 channels * 2 * 2 spatial = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # Changed from 256 * 4 * 4
        self.fc2 = nn.Linear(512, 64)
        
        # Separate heads for different action types (steering, gas, brake)
        """
        The Car Racing action space is asymmetric:
        Steering: [-1, 1] (tanh works perfectly)
        Gas: [0, 1] (sigmoid)
        Brake: [0, 1] (sigmoid)
        """
        self.mean_head = nn.Linear(64, 3)
        self.log_std_head = nn.Linear(64, 3)

        # Constants for log_std clamping
        # This is used to prevent the log_std from becoming too large or too small
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        self.register_buffer('action_scale', torch.FloatTensor([1.0, 0.5, 0.5]))
        self.register_buffer('action_bias', torch.FloatTensor([0.0, 0.5, 0.5]))

    def forward(self, x):
        # CNN processing
        x = self.convnet(x)
<<<<<<< Updated upstream
        x = x.reshape(x.size(0), -1)  # Flatten
=======
        x = x.reshape(x.size(0), -1)
>>>>>>> Stashed changes
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def step(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # Unbounded sample

        # Squash to [-1, 1]
        tanh_action = torch.tanh(u)

        # Log prob with tanh correction (critical!)
        log_prob = dist.log_prob(u).sum(dim=1)
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6).sum(dim=1)

        # Scale to actual action bounds
        action = torch.cat([
            tanh_action[:, 0:1],  # Steering: [-1, 1]
            (tanh_action[:, 1:2] + 1) / 2,  # Gas: [0, 1]
            (tanh_action[:, 2:3] + 1) / 2,  # Brake: [0, 1]
        ], dim=1)

        return action, log_prob
    
    def step_new(self, state):
        """SAC action sampling with tanh squashing and log-prob correction"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        # Reparameterization trick (mean + std * N(0,1))
        # This allows backprop with respect to mean and std
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


    def act(self, state):
        with torch.no_grad():
            # We can use step but ignore log_prob
            action, _ = self.step(state)
            return action.cpu().numpy()

    def infer(self, state):
        with torch.no_grad():
            means, _ = self.forward(state)
            return means.cpu().numpy()

    def get_log_prob(self, states, actions):
        # states, actions are taken from experiecne replay buffer and get_log_prob computes what log probability the current policy would give those states and actions 
        # Computes log π(a|s): Allows recomputing log_prob with gradients during policy updates
        # Part of the policy gradient: ∇θ J(θ) = E[∇θ log π(a|s) * A]
        # Enables backpropagation (gradient can flow): get_log_prob -> policy_network -> policy_loss -> optimizer.step()

        # Step 1: Forward pass  
        means, log_stds = self.forward(states) # means:(batch_size, 3) = [[steering_mean, gas_mean, brake_mean]], log_stds:(batch_size, 3) = [[steering_log_std, gas_log_std, brake_log_std]]
        
        # Step 2: Convert log_std to std
        stds = torch.exp(log_stds) # stds:(batch_size, 3) = [[steering_std, gas_std, brake_std]]
        
        # Step 3: Create Gaussian distribution
        dist = torch.distributions.Normal(means, stds) # dist: Gaussian distribution with mean and std
        
        # Step 4: Calculate log probability
        # Important: sum log_probs across the action dimensions
        log_probs = dist.log_prob(actions).sum(dim=1) # log_probs:(batch_size,) = [total_log_prob] = [log_prob_steering + log_prob_gas + log_prob_brake] = [log(Normal(0.2, 0.223).pdf(steering_raw)) + log(Normal(0.7, 0.135).pdf(gas_raw)) + log(Normal(0.1, 0.050).pdf(brake_raw))]
        
        return log_probs


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet()
        # ConvNet output: 256 channels * 2 * 2 spatial = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # Changed from 256 * 4 * 4
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.convnet(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # Flatten to (batch,)

class QNetwork(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        
        # Use the same ConvNet as PolicyNetwork/ValueNetwork
        self.convnet = ConvNet()
        
        # ConvNet output: 256 * 2 * 2 = 1024
        self.flatten_size = 256 * 2 * 2
        
        # Fully connected layers (state features + action)
        self.fc1 = nn.Linear(self.flatten_size + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, state, action):
        """Takes both state and action"""
        # Normalize state
        state = state.float() / 255.0
        
        # Process through ConvNet
        x = self.convnet(state)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 1024)
        
        # Concatenate with action
        x = torch.cat([x, action], dim=1)  # (batch, 1024 + action_dim)
        
        x = torch.relu(self.fc1(x))
<<<<<<< Updated upstream
        q_value = self.fc2(x)
        return q_value
=======
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
>>>>>>> Stashed changes

class ConvNet(nn.Module):
    """Lighter CNN for faster training"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (4, 84, 84) -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32, 20, 20) -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 9, 9) -> (128, 7, 7)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # (128, 7, 7) -> (256, 2, 2)
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)