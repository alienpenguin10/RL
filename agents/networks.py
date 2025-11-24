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
        # This is used to prevent the log_std from becoming too large or too small
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
        log_prob = dist.log_prob(action).sum(dim=1) # log_prob:(batch_size,) = [total_log_prob] = [log_prob_steering + log_prob_gas + log_prob_brake] = [log(Normal(0.2, 0.223).pdf(steering_raw)) + log(Normal(0.7, 0.135).pdf(gas_raw)) + log(Normal(0.1, 0.050).pdf(brake_raw))]
        
        # Step 5: Clamp actions to valid environment ranges
        # action[0] = steering ∈ [-1, 1]    (left/right turn)
        # action[1] = gas ∈ [0, 1]          (0=no gas, 1=full gas)
        # action[2] = brake ∈ [0, 1]        (0=no brake, 1=full brake)
        action[:, 0] = torch.clamp(action[:, 0], -1, 1)   # Steering: [-1, 1]
        action[:, 1] = torch.clamp(action[:, 1], 0, 1)    # Gas: [0, 1]
        action[:, 2] = torch.clamp(action[:, 2], 0, 1)    # Brake: [0, 1]

        # action:(batch_size, 3) = [[steering, gas, brake]]
        # log_prob:(batch_size,) = [total_log_prob]
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
