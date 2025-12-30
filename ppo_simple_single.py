from pyexpat import features
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import gymnasium as gym
# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training will continue without logging")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        print(f"Input shape: {input_shape}; Output shape: {output_shape}")
        # Shared CNN feature extractor for 84x84x4 input
        # Backbone: Nature CNN (scaled for 96x96 input)
        # We use ReLU as it is standard for vision backbones and empirically faster/better than Tanh.
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)), # 84→20
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), # 20→9
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), # 9→7
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), # 3136 → 512
            nn.ReLU()
        )   
        # Using Orthogonal Initialization
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)
        
        # Actor Head: Beta Distribution Parameters
        # We need alpha and beta for each of the 3 actions (Steering, Gas, Brake).
        # Total outputs = 3 actions * 2 params = 6.
        self.actor_head = layer_init(nn.Linear(512, 6), std=0.01)
    
    def get_value(self, state):
        features = self.network(state)
        return self.critic(features)

    def get_action(self, state):
        # Takes a single state -> samples a new action from policy dist
        features = self.network(state)

        # Get Alpha and Beta parameters
        # We use Softplus + 1.0 to ensure alpha, beta >= 1.0.
        # This constrains the Beta distribution to be unimodal (bell-shaped),
        # preventing the "U-shaped" bimodality that destabilizes training.
        policy_out = self.actor_head(features)
        alpha_beta = torch.nn.functional.softplus(policy_out) + 1.0

        # Split into alpha and beta components (B, 3) each
        alpha, beta = torch.chunk(alpha_beta, 2, dim=-1)

        # Create Beta Distribution
        dist = torch.distributions.Beta(alpha, beta)

        # Sample raw actions in range
        raw_action = dist.sample()

        # Calculate Log Prob (sum over the 3 action dimensions)
        action_log_probs_per_dim = dist.log_prob(raw_action)

        # Affine Transformation
        # Map raw sample to environment bounds
        # Steering (idx 0): -> [-1, 1] via y = x*2-1
        # Gas (idx 1): -> [0, 1] via y = x
        # Brake (idx 2): -> [0, 1] via y = x
        action = torch.stack([
            raw_action[:, 0] * 2 - 1, # Steering
            raw_action[:, 1], # Gas
            raw_action[:, 2] # Brake
        ], dim=1)

        # Log Prob Correction
        # When transforming a variable, we must correct the density.
        # For steering y = 2x - 1, dy/dx = 2.
        # log_prob(y) = log_prob(x) - log(|dy/dx|) = log_prob(x) - log(2)
        # This correction applies only to the steering dimension.
        action_log_probs_per_dim[:, 0] -= torch.log(torch.tensor(2.0))
        action_log_prob = action_log_probs_per_dim.sum(dim=1)
        value = self.critic(features)
        # Return tensors (keep on GPU) - squeeze to remove batch dimension
        return action.squeeze(0), action_log_prob.squeeze(0), value.squeeze(0)
    
    def evaluate(self, states, actions):
        # takes in batch of states and actions -> doesn't sample evaluates the log prob of specific action under the current policy
        # also returns entropy regularization term
        features = self.network(states)

        # Reconstruct Distribution
        policy_out = self.actor_head(features) # [B, 6]
        alpha_beta = torch.nn.functional.softplus(policy_out) + 1.0 # [B, 6]
        alpha, beta = torch.chunk(alpha_beta, 2, dim=-1) # [B, 3], [B, 3]
        dist = torch.distributions.Beta(alpha, beta) # [B, 3]

        # Inverse Transformation
        # The 'actions' passed here are from the reply buffer (Env space)
        # We must map them back to evaluate them under the Beta distribution
        # Inverse steering: [-1, 1] -> [0, 1]:  y = x*2-1 -> x = (y + 1) / 2
        # Inverse gas: [0, 1] -> [0, 1]: y = x
        # Inverse brake: [0, 1] -> [0, 1]: y = x
        raw_actions = torch.stack([
            (actions[:, 0] + 1) / 2, # Steering
            actions[:, 1], # Gas
            actions[:, 2] # Brake
        ], dim=1)
        # Numerical stability: clamp to avoid exact 0 or 1 which can cause inf log_prob
        raw_actions = torch.clamp(raw_actions, 1e-6, 1.0 - 1e-6)

        log_prob_per_dim = dist.log_prob(raw_actions)
        entropy = dist.entropy().sum(dim=1)
        # Apply the same correction for the steering dimension
        log_prob_per_dim[:, 0] -= torch.log(torch.tensor(2.0)) 
        log_prob = log_prob_per_dim.sum(dim=1)
        
        value = self.critic(features)
        return log_prob, value, entropy

class PPOAgent:
    def __init__(self, env):
        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.policy = ActorCritic(input_shape=self.num_observations, output_shape=self.num_actions).to(DEVICE)
        self.optimizer = Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_UPDATES)
    
    def update(self, rollouts):
        # rollouts: {'states': states, 'actions': actions, 'returns': returns, 'advantages': advantages, 'values': values, 'log_probs': log_probs}
        states = rollouts['states']
        actions = rollouts['actions'] 
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        values = rollouts['values']
        old_log_probs = rollouts['log_probs']

        # Normalize advantages
        normalised_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, normalised_advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(NUM_EPOCHS):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)
                
                # Policy Loss Formula
                # r = log(pi(a|s)) / log_old(pi(a|s))
                # L_clip = min(r * A, clip(r, 1-eps, 1+eps) * A)
                # Policy Loss = - E[L_clip]
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - EPSILONS, 1 + EPSILONS) * b_advantages
                policy_loss = - torch.min(surr1, surr2).mean()

                # Value Loss Formula
                # Value Loss = 1/2 E[(V(s) - V(s'))^2]
                b_returns = b_returns.reshape(-1)
                b_values = values.reshape(-1)
                value_loss = 0.5 * (b_returns - b_values).pow(2).mean()

                # Policy Entropy = E[log(pi(a|s))]
                entropy_loss = entropy.mean()

                # Total Loss = Policy Loss - ENTROPY_COEFF * Policy Entropy + VALUE_COEFF * Value Loss
                total_loss = policy_loss - ENTROPY_COEFF * entropy_loss  + VALUE_COEFF * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            
        # self.lr_scheduler.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
        }

def compute_gae(rewards, values, terminateds, truncateds, next_value):
    #TD Error = r + gamma * V(s_{t+1}) - V(s_t)
    # A_t = TD Error + gamma * lambda * A(s_{t+1})
    # Recall returns can be computed in two different ways:
    # 1. Monte Carlo returns: G_t = gamma^k * r_t + gamma^(k-1) * r_(t+1) + ... + gamma * r_(t+k-1)
    # 2. GAE returns: G_t = A_t + V(s_t) since A_t = G_t - V(s_t)
    # returns: Uses returns as targets to train the critic function to predict better state, value predictions.
    # terminated: bootstrap_mask=0, gae_mask=0
    # truncated: bootstrap_mask=1, gae_mask=1  
    # (i.e., we DO bootstrap and accumulate for truncated episodes)
    advantages = [] # Uses this to determine which actions were better than expected, helping the policy improve.
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # This is the last step of rollout; Only mask terminated states, not truncated ones
            next_non_terminal = 1.0 - terminateds[t]
            next_values = next_value
        else:
            # Use the NEXT transition's terminated flag
            next_non_terminal = 1.0 - terminateds[t + 1]
            next_values = values[t + 1]

        # TD error with proper masking
        delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
        
        # GAE accumulation
        # If terminated (episode ended naturally), don't accumulate future advantages
        # If truncated (episode ended due to time limit),DO accumulate (we bootstrapped above)
        terminated_mask = terminateds[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - terminated_mask) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(returns).to(DEVICE), torch.tensor(advantages).to(DEVICE)

TOTAL_TIMESTEPS = 2000000
HORIZON = 2048 # One episode is 200 steps for car racing
NUM_UPDATES = int(TOTAL_TIMESTEPS / HORIZON) # 2000000 / 2048 = 976
NUM_EPOCHS = 4
NUM_MINIBATCHES = 16 
BATCH_SIZE = HORIZON // NUM_MINIBATCHES # 2048 // 16 = 128

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILONS = 0.1
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env_name='CarRacing-v3'):

    """ Hyperparameters """

    # Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name=f"ppo_{env_name}",
            config={
                "algorithm": "ppo",
                "environment": env_name,
                "max_timesteps": TOTAL_TIMESTEPS,
                "buffer_size": HORIZON,
                "mini_batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
            }
        )
    # Import frame stacking wrapper
    from wrappers import PreprocessWrapper, FrameStack 
    
    env = gym.make('CarRacing-v3', continuous=True)
    env = PreprocessWrapper(env)
    env = FrameStack(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = PPOAgent(env)

    state, _ = env.reset()
    state = torch.Tensor(state).to(DEVICE)
    total_steps = 0
    update_count = 0
    avg_reward = 0
    episode_rewards = []
    
    while total_steps < TOTAL_TIMESTEPS:
        states = torch.zeros((HORIZON, *env.observation_space.shape)).to(DEVICE)
        actions = torch.zeros((HORIZON, *env.action_space.shape)).to(DEVICE)
        rewards = torch.zeros((HORIZON)).to(DEVICE)
        terminateds = torch.zeros((HORIZON)).to(DEVICE)
        truncateds = torch.zeros((HORIZON)).to(DEVICE)
        values = torch.zeros((HORIZON)).to(DEVICE)
        log_probs = torch.zeros((HORIZON)).to(DEVICE)

        rollout_rewards = []

        # Rollout
        for step in range(HORIZON):
            states[step] = state
            with torch.no_grad():
                # CNN expects input with a batch dimension: (batch_size, channels, height, width)
                # without .unsqueeze(0), state.shape = (channels, height, width)
                # with .unsqueeze(0), state.shape = (1, channels, height, width)
                action_tensor, log_prob_tensor, value_tensor = agent.policy.get_action(state.unsqueeze(0))
                # Keep tensors on GPU - no conversion needed
                values[step] = value_tensor
                actions[step] = action_tensor
                log_probs[step] = log_prob_tensor

            # Only convert to NumPy for environment interaction
            raw_action = action_tensor.cpu().numpy()
            next_state, reward, terminated, truncated, info = env.step(raw_action)
            if "episode" in info:
                print(f"Global Step: {total_steps}, Episode Return: {info['episode']['r']}, Length: {info['episode']['l']}")
                episode_rewards.append(info['episode']['r'])
            rollout_rewards.append(reward)

            # Crucial: Only treat as 'done' if terminated (failure), not truncated (time limit)
            terminateds[step] = terminated
            truncateds[step] = truncated
            rewards[step] = reward
            if terminated or truncated:
                next_state, _ = env.reset()
            state = torch.Tensor(next_state).to(DEVICE)
            total_steps += 1
        
        # Boostrap value if not done
        with torch.no_grad():
            next_value = agent.policy.get_value(state.unsqueeze(0)).reshape(-1)

        returns, advantages = compute_gae(rewards, values, terminateds, truncateds, next_value)
        
        rollouts = {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'values': values,
            'log_probs': log_probs,
        }

        update_metrics = agent.update(rollouts)
        update_count += 1

        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f'Average Reward of last 10 episode is {update_count}: {avg_reward}')
        average_rollout_reward = np.mean(rollout_rewards)
        print(f"Rollout Average Reward: {average_rollout_reward}")
     

        log_dict = {
            "policy_loss": update_metrics['policy_loss'],
            "value_function_loss": update_metrics['value_loss'],
            "total_loss": update_metrics['total_loss'],
            "average_episode_reward": avg_reward,
            "average_rollout_reward": average_rollout_reward,
        }

        if WANDB_AVAILABLE:
            wandb.log(log_dict)

if __name__ == '__main__':
    env_name = 'CarRacing-v3'
    train(env_name=env_name)