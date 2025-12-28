import numpy as np
import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, num_frames, output_shape):
        super().__init__()
        # Using Orthogonal Initialization
        self.frame_stacking = num_frames > 1
        self.conv_output_size = 32*8*8  # Output size from ConvNet = 2048

        self.conv1 = nn.Conv2d(in_channels=num_frames, out_channels=16, kernel_size=7, stride=4, padding=(8,2)) # (B,16,24,24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B,16,12,12)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # (B,32,8,8)
        self.fc1 = layer_init(nn.Linear(self.conv_output_size, 256))
        self.fc2 = layer_init(nn.Linear(256, 128))
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

        self.steer = layer_init(nn.Linear(128, 1), std=0.01)
        self.speed = layer_init(nn.Linear(128, 1), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 1))  # Shared log std for both actions
        
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
    
    def forward(self, x):
        assert isinstance(x, torch.FloatTensor), "Input must be a FloatTensor"

        x = x.float() / 255.0 # Normalize input
        if len(x.shape) == 3:
            x = x.unsqueeze(0) # To make sure state has a batch dimension
        if (not self.frame_stacking):
            x = x.permute(0, 3, 1, 2) # Change from (B, H, W, C) to (B, C, H, W)
        
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act1(self.conv2(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = self.act2(self.fc1(x))
        x = self.act2(self.fc2(x))

        steer_mean = self.act2(self.steer(x))
        speed_mean = self.act2(self.speed(x))
        steer_std = torch.exp(self.actor_logstd.expand_as(steer_mean))
        speed_std = torch.exp(self.actor_logstd.expand_as(speed_mean))

        value = self.act2(self.critic(x))

        return steer_mean, steer_std, speed_mean, speed_std, value
    
    def get_action(self, state):
        # Takes a single state -> samples a new action from policy dist
        steer_mean, steer_std, speed_mean, speed_std, value = self.forward(state)
        steer_dist = torch.distributions.Normal(steer_mean, steer_std)
        speed_dist = torch.distributions.Normal(speed_mean, speed_std)
        steer_action = steer_dist.sample()
        speed_action = speed_dist.sample()
        steer = steer_action.cpu().numpy().flatten()
        speed = speed_action.cpu().numpy().flatten()
        steer_log_prob = steer_dist.log_prob(steer_action).sum(1).cpu().numpy().flatten()
        speed_log_prob = speed_dist.log_prob(speed_action).sum(1).cpu().numpy().flatten()

        action = np.array([steer[0], speed[0]])
        log_prob = steer_log_prob + speed_log_prob

        return action, log_prob, value
    
    def get_value(self, state):
        # Takes a single state -> returns value estimate
        _, _, _, _, value = self.forward(state)
        return value

    def evaluate(self, states, actions):
        # takes in batch of states and actions -> doesn't sample evaluates the log prob of specific action under the current policy
        # also returns entropy regularization term
        steer_mean, steer_std, speed_mean, speed_std, value = self.forward(states)
        steer_dist = torch.distributions.Normal(steer_mean, steer_std)
        speed_dist = torch.distributions.Normal(speed_mean, speed_std)
        steer_actions = actions[:, 0].unsqueeze(1)
        speed_actions = actions[:, 1].unsqueeze(1)
        steer_log_prob = steer_dist.log_prob(steer_actions).sum(1)
        speed_log_prob = speed_dist.log_prob(speed_actions).sum(1)
        log_prob = steer_log_prob + speed_log_prob
        entropy = steer_dist.entropy().sum(1) + speed_dist.entropy().sum(1)
        
        return log_prob, value, entropy