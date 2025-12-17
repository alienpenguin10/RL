"""
PolicyNetwork(nn.Module) and ValueNetwork(nn.Module) for VPG
Same architecture as REINFORCE, adapted for VPG usage
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

# Remove global device check, handled in Agent

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet_StackedFrames(num_frames=4)
        # ConvNet output: 256 channels * 4 * 4 spatial = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        
        # Separate heads for different action types (steering, gas, brake)
        """
        The Car Racing action space is asymmetric:
        Steering: [-1, 1] (tanh works perfectly)
        Gas: [0, 1] (tanh) but clamped later
        Brake: [0, 1] (tanh)
        """
        self.steering_mean = nn.Linear(64, 1)  
        self.steering_log_std = nn.Linear(64, 1)  

        self.gas_mean = nn.Linear(64, 1)  
        self.gas_log_std = nn.Linear(64, 1) 

        self.brake_mean = nn.Linear(64, 1)  
        self.brake_log_std = nn.Linear(64, 1) 

        # Weights
        torch.nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        torch.nn.init.orthogonal_(self.steering_mean.weight, 0.01)
        torch.nn.init.orthogonal_(self.gas_mean.weight, 0.01)
        torch.nn.init.orthogonal_(self.brake_mean.weight, 0.01)
        
        # Biases
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.steering_mean.bias.data.fill_(0)

        # CRITICAL: Initialize Gas/Brake bias to -2.0
        # Tanh(-2.0) = -0.96. 
        # RemapWrapper maps -0.96 -> ~0.02 (almost 0 gas/brake)
        # This prevents the "driving with parking brake on" problem.
        self.gas_mean.bias.data.fill_(-2.0)
        self.brake_mean.bias.data.fill_(-2.0)
        
        self.LOG_STD_MIN = -3.0
        self.LOG_STD_MAX = 0.0

    def forward(self, x):
        # Normalize pixel values to [0, 1]
        # x = x.float() / 255.0 # Done inside ConvNet_StackedFrames now

        # CNN processing
        x = self.convnet(x)
        x = x.reshape(x.size(0), -1)  # Flatten, prev .view() didn't work, errored saying the data is contiguous
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # ALL heads use Tanh
        # steering_mean = torch.tanh(self.steering_mean(x))
        # gas_mean = torch.tanh(self.gas_mean(x))
        # brake_mean = torch.tanh(self.brake_mean(x))
        steering_mean = self.steering_mean(x) 
        gas_mean = self.gas_mean(x)
        brake_mean = self.brake_mean(x)

        # Policy Log Stds
        steering_log_std = torch.clamp(self.steering_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        gas_log_std = torch.clamp(self.gas_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        brake_log_std = torch.clamp(self.brake_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Concatenate means and log stds
        means = torch.cat([steering_mean, gas_mean, brake_mean], dim=1)
        log_stds = torch.cat([steering_log_std, gas_log_std, brake_log_std], dim=1)

        return means, log_stds

    def step(self, state):
        # Step 1: Forward pass
        # Input: state shape (batch_size=1, 3, 96, 96) - RGB image
        # State is already a tensor on device from agent.preprocess_state
        means, log_stds = self.forward(state) # means:(batch_size, 3) = [[steering_mean, gas_mean, brake_mean]], log_stds:(batch_size, 3) = [[steering_log_std, gas_log_std, brake_log_std]]
        
        # Step 2: Convert log_std to std
        stds = torch.exp(log_stds) # stds:(batch_size, 3) = [[steering_std, gas_std, brake_std]]
        
        # Step 3: Create Gaussian distribution
        # Creates 3 independent Normal distributions:
        #    Steering: Normal(mean=0.2, std=0.223) 
        #    Gas:      Normal(mean=0.7, std=0.135)  
        #    Brake:    Normal(mean=0.1, std=0.050)
        dist = torch.distributions.Normal(means, stds)
        action = dist.sample() # action:(batch_size, 3) = [[steering_raw, gas_raw, brake_raw]]
        
        # Step 4: Calculate log probability (BEFORE clamping)
        # Important: Calculate log_prob using original sampled values, not clamped ones!
        # This ensures the probability math matches the distribution we sampled from
        # Needed in loss function calculation
        log_prob = dist.log_prob(action).sum(dim=1) # log_prob:(batch_size,) = [total_log_prob] = [log_prob_steering + log_prob_gas + log_prob_brake] = [log(Normal(0.2, 0.223).pdf(steering_raw)) + log(Normal(0.7, 0.135).pdf(gas_raw)) + log(Normal(0.1, 0.050).pdf(brake_raw))]
        
        # Step 5: Clamp actions to valid environment ranges
        # action[0] = steering ∈ [-1, 1]    (left/right turn)
        # action[1] = gas ∈ [-1, 1]         (remapped later)
        # action[2] = brake ∈ [-1, 1]       (remapped later)
        #action[:, 0] = torch.clamp(action[:, 0], -1, 1)   # Steering: [-1, 1]
        #action[:, 1] = torch.clamp(action[:, 1], -1, 1)   # Gas: [-1, 1]
        #action[:, 2] = torch.clamp(action[:, 2], -1, 1)   # Brake: [-1, 1]

        # action:(batch_size, 3) = [[steering, gas, brake]]
        # log_prob:(batch_size,) = [total_log_prob]
        # Return the raw action [-inf, inf]
        return action, log_prob
        

    def step_new(self, state):
        """SAC action sampling with tanh squashing and log-prob correction"""
        # Forward pass
        mu, logstd = self.forward(state)
        std = logstd.exp()
        
        # Create distribution and sample with reparameterization
        dist = Normal(mu, std)
        actions = dist.rsample()  # Reparameterization trick for gradients
        batch_size = actions.shape[0]
        
        # Log probability before squashing
        log_prob = dist.log_prob(actions).sum(dim=-1)
        
        # Apply tanh squashing
        squashed = torch.tanh(actions)
        
        # Log-prob correction for tanh squashing (more stable formula)
        # Corrects for the change of variables: log π(a) = log μ(u) - log|da/du|
        log_prob_correction = torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=1)
        log_prob = log_prob - log_prob_correction
        
        # Scale actions to environment ranges
        steering_action = squashed[:, 0].view(batch_size, 1)  # [-1, 1]
        gas_action = ((squashed[:, 1] + 1) / 2).view(batch_size, 1)  # [0, 1]
        brake_action = ((squashed[:, 2] + 1) / 2).view(batch_size, 1)  # [0, 1]
        
        # Clamp brake to valid range
        brake_action = torch.clamp(brake_action, 0.0, 1.0)
        
        # Concatenate final action
        action = torch.cat((steering_action, gas_action, brake_action), dim=1)
        
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

    def get_action_and_log_prob(self, state, action=None):
        """
        Computes action and log_prob handling the Tanh transformation correctly.
        """
        means, log_stds = self.forward(state)
        stds = torch.exp(log_stds)
        
        # Base distribution (Gaussian)
        normal = Normal(means, stds)
        
        if action is None:
            # Sampling phase
            raw_action = normal.rsample() # Reparameterization trick
        else:
            # Training phase: We have the squashed action, we need to recover raw?
            # Actually, standard PPO practice with Tanh is usually to store the 
            # RAW (unsquashed) action in the buffer to make log_prob calc easier.
            # But since we stored squashed or raw? 
            # Let's assume we pass RAW actions here (see ppo.py update below).
            raw_action = action

        # Squash action
        squashed_action = torch.tanh(raw_action)
        
        # --- FIX 2: Tanh Log-Prob Correction ---
        # log(pi(a)) = log(pi(u)) - sum(log(1 - tanh(u)^2))
        log_prob = normal.log_prob(raw_action).sum(dim=1)
        correction = (2 * (np.log(2) - raw_action - torch.nn.functional.softplus(-2 * raw_action))).sum(dim=1)
        log_prob -= correction

        return squashed_action, raw_action, log_prob

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet_StackedFrames(num_frames=4)
        # ConvNet output: 256 channels * 4 * 4 spatial = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x = x.float() / 255.0 # Done inside ConvNet_StackedFrames
        x = self.convnet(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # Flatten to (batch,)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet()

        self.fc1 = nn.Linear(256 * 4 * 4 + 3, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = self.convnet(state)
        x = x.reshape(x.size(0), -1)

        x = torch.cat([x, action], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for processing 96*96*3 RGB images
        # Receptive field on final layer = 88 x 88 pixels

        # spatial resolution (height/width): 96 -> 24 -> 12 -> 10 -> 8 -> 6 -> 4
        # channels: 3 -> 16 -> 32 -> 64 -> 128 -> 256
        # Input: (B,3,96,96)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=4, padding=2) # (B,16,24,24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B,16,12,12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1) # (B,32,10,10)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) # (B,64,8,8)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1) # (B,128,6,6)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1) # (B,256,4,4)

    def forward(self, x):
        x = x.float() / 255.0 # Normalize pixel values to [0, 1]
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        return x
    
class ConvNet_StackedFrames(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        # CNN for processing stacked grayscale frames of shape (num_frames, 84, 96)

        # Spatial resolution (height/width): 84 -> 24 -> 12 -> 10 -> 8 -> 6 -> 4
        # channels: num_frames -> 16 -> 32 -> 64 -> 128 -> 256 

        # Input: (B,num_frames,84,96)
        # asymmetric padding p=(8,2) adds 84+16 = 100 to heignt and 96+4 = 100 to width
        self.conv1 = nn.Conv2d(in_channels=num_frames, out_channels=16, kernel_size=7, stride=4, padding=(8,2)) # (B,16,24,24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B,16,12,12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1) # (B,32,10,10)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) # (B,64,8,8)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1) # (B,128,6,6)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1) # (B,256,4,4)

    def forward(self, x):
        x = x.float() / 255.0 # Normalize pixel values to [0, 1]
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        return x