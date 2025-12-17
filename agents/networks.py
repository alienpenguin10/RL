"""
PolicyNetwork(nn.Module) and ValueNetwork(nn.Module) for VPG
Same architecture as REINFORCE, adapted for VPG usage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), activation()])
        layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ConvNet(nn.Module):
    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()
        # obs_shape from gym: (H, W, C) -> convert to (C, H, W) for PyTorch conv
        if len(obs_shape) == 3:
            # Assume gym format (H, W, C) if last dim is small (channels)
            if obs_shape[2] <= 4:
                self.input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
            else:
                self.input_shape = obs_shape  # Already (C, H, W)
        else:
            self.input_shape = obs_shape
            
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            conv_out_size = dummy.numel()
        
        self.fc = nn.Linear(conv_out_size, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # Check if input is (B, H, W, C) and needs permutation to (B, C, H, W)
        # We use self.input_shape[0] which stores the expected channel count
        if len(x.shape) == 4:
            if x.shape[1] != self.input_shape[0] and x.shape[-1] == self.input_shape[0]:
                x = x.permute(0, 3, 1, 2).contiguous()
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))
    
class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256], log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.cnn = ConvNet(obs_shape, feature_dim)
        self.mean_net = MLP(feature_dim, hidden_dims, action_dim)
        self.log_std_net = MLP(feature_dim, hidden_dims, action_dim)
        self.action_dim = action_dim

    def forward(self, x):
        output = self.cnn(x)
        mean = self.mean_net(output)
        log_std = torch.clamp(self.log_std_net(output), self.log_std_min, self.log_std_max)
        return mean, log_std

    def step(self, state, deterministic=False):

        means, log_stds = self.forward(state)
        stds = torch.exp(log_stds)

        if deterministic:
            return torch.tanh(means), None, None

        normal = torch.distributions.Normal(means, stds)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)
        
        return action, log_prob, entropy


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
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        self.cnn = ConvNet(obs_shape, feature_dim)
        self.net = MLP(feature_dim + action_dim, hidden_dims, 1)
    
    def forward(self, state, action):
        """Takes both state and action"""
        features = self.cnn(state)
        return self.net(torch.cat([features, action], dim=-1))
    

# class QNetwork(nn.Module):
#     def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256]):
#         super().__init__()
        
#         # Use the same ConvNet as PolicyNetwork/ValueNetwork
#         self.convnet = ConvNet(obs_shape, feature_dim)
#         self.net = MLP(feature_dim + action_dim, hidden_dims, 1)
        
#         # ConvNet output: 256 * 2 * 2 = 1024
#         self.flatten_size = 256 * 2 * 2
        
#         # Fully connected layers (state features + action)
#         self.fc1 = nn.Linear(self.flatten_size + action_dim, 512)
#         self.fc2 = nn.Linear(512, 1)
    
#     def forward(self, state, action):
#         """Takes both state and action"""
#         # Normalize state
#         state = state.float() / 255.0
        
#         # Process through ConvNet
#         x = self.convnet(state)
#         x = x.view(x.size(0), -1)  # Flatten: (batch, 1024)
        
#         # Concatenate with action
#         x = torch.cat([x, action], dim=1)  # (batch, 1024 + action_dim)
        
#         x = torch.relu(self.fc1(x))
#         q_value = self.fc2(x)
#         return q_value