"""
PolicyNetwork(nn.Module) and ValueNetwork(nn.Module) for VPG
Same architecture as REINFORCE, adapted for VPG usage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))
    
class PolicyNetwork(nn.Module):

    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256], log_std_min=-20, log_std_max=2):
        super().__init__()
        
        # Constants for log_std clamping
        # This is used to prevent the log_std from becoming too large or too small
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.cnn = ConvNet(obs_shape, feature_dim)
        self.mean_net = MultiLayerPerceptron(feature_dim, hidden_dims, action_dim)
        self.log_std_net = MultiLayerPerceptron(feature_dim, hidden_dims, action_dim)
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

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = ConvNet()
        # ConvNet output: 256 channels * 2 * 2 spatial = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # Changed from 256 * 4 * 4
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
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        self.cnn = ConvNet(obs_shape, feature_dim)
        self.net = MultiLayerPerceptron(feature_dim + action_dim, hidden_dims, 1)
    
    def forward(self, state, action):
        features = self.cnn(state)
        return self.net(torch.cat([features, action], dim=-1))
