from pyexpat import features
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import gymnasium as gym
from pprint import pprint
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

# Environment factory for vectorized environments
def make_env(seed):
    """Factory function to create a wrapped CarRacing environment."""
    def thunk():
        from wrappers import PreprocessWrapper, FrameStack
        env = gym.make('CarRacing-v3', continuous=True)
        env = PreprocessWrapper(env)
        env = FrameStack(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


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
        return action.cpu().numpy().flatten(), action_log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()
    
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
    def __init__(self, observation_space, action_space):
        self.num_observations = observation_space.shape[0]
        self.num_actions = action_space.shape[0]
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

def compute_gae(rewards, values, terminated, terminateds, next_value):
    #TD Error = r + gamma * V(s_{t+1}) - V(s_t)
    # A_t = TD Error + gamma * lambda * A(s_{t+1})
    # Recall returns can be computed in two different ways:
    # 1. Monte Carlo returns: G_t = gamma^k * r_t + gamma^(k-1) * r_(t+1) + ... + gamma * r_(t+k-1)
    # 2. GAE returns: G_t = A_t + V(s_t) since A_t = G_t - V(s_t)
    # returns: Uses returns as targets to train the critic function to predict better state, value predictions.
    advantages = [] # Uses this to determine which actions were better than expected, helping the policy improve.
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - terminated
            next_values = next_value
        else:
            next_non_terminal = 1.0 - terminateds[t + 1]
            next_values = values[t + 1]
        
        delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(returns).to(DEVICE), torch.tensor(advantages).to(DEVICE)

# Vectorized Environment Configuration
NUM_ENVS = 48  # Number of parallel environments
STEPS_PER_ENV = 128  # Steps per environment per rollout
TOTAL_STEPS = NUM_ENVS * STEPS_PER_ENV  # 48 * 128 = 6144 total steps per rollout

# Training Configuration
TOTAL_TIMESTEPS = 2_000_000
NUM_UPDATES = TOTAL_TIMESTEPS // TOTAL_STEPS  # 2000000 // 6144 = ~326 updates
NUM_EPOCHS = 4
NUM_MINIBATCHES = 32
BATCH_SIZE = TOTAL_STEPS // NUM_MINIBATCHES  # 6144 // 32 = 192

# PPO Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILONS = 0.1
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env_name='CarRacing-v3'):

    """Vectorized PPO Training"""

    # Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name=f"ppo_vectorized_{env_name}",
            config={
                "algorithm": "ppo",
                "environment": env_name,
                "num_envs": NUM_ENVS,
                "max_timesteps": TOTAL_TIMESTEPS,
                "steps_per_env": STEPS_PER_ENV,
                "total_steps_per_rollout": TOTAL_STEPS,
                "mini_batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
            }
        )
    
    # Create vectorized environments
    print(f"[1/4] Creating {NUM_ENVS} parallel environments...")
    import time
    start_time = time.time()
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])
    print(f"[2/4] Environments created in {time.time() - start_time:.2f}s")
    
    # Create agent using single env's observation/action spaces
    print(f"[3/4] Creating PPO agent on {DEVICE}...")
    agent = PPOAgent(envs.single_observation_space, envs.single_action_space)
    print(f"[4/4] Agent created. Resetting environments...")

    # Reset all environments
    states, _ = envs.reset(seed=42)
    states = torch.Tensor(states).to(DEVICE)  # Shape: (NUM_ENVS, 4, 84, 84)
    print(f"✓ Initialization complete! Starting training with {NUM_ENVS} parallel environments")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,} | Updates: {NUM_UPDATES} | Steps per rollout: {TOTAL_STEPS}")
    print("-" * 80)

    total_steps = 0
    update_count = 0
    episode_rewards = []
    
    # Manual episode tracking for each environment
    episode_reward_tracker = np.zeros(NUM_ENVS)
    episode_length_tracker = np.zeros(NUM_ENVS, dtype=int)
    
    while total_steps < TOTAL_TIMESTEPS:
        print(f"[Update {update_count + 1}/{NUM_UPDATES}] Collecting rollout... (Steps: {total_steps:,}/{TOTAL_TIMESTEPS:,})", end='', flush=True)
        
        # Track episodes that complete during this rollout
        rollout_episode_rewards = []
        
        # Allocate storage for rollout (shape: [steps, num_envs, ...])
        obs_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS, *envs.single_observation_space.shape)).to(DEVICE)
        actions_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS, *envs.single_action_space.shape)).to(DEVICE)
        rewards_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)
        dones_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)
        values_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)
        log_probs_buffer = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)

        # Rollout across all environments
        for step in range(STEPS_PER_ENV):
            obs_buffer[step] = states
            
            with torch.no_grad():
                # Forward pass for all environments at once
                # states already has batch dimension: (NUM_ENVS, 4, 84, 84)
                raw_actions, log_probs, values = agent.policy.get_action(states)
                
                # Reshape outputs from flattened to (NUM_ENVS, action_dim)
                raw_actions = raw_actions.reshape(NUM_ENVS, -1)
                log_probs = log_probs.reshape(NUM_ENVS)
                values = values.reshape(NUM_ENVS)
                
            actions_buffer[step] = torch.tensor(raw_actions).to(DEVICE)
            log_probs_buffer[step] = torch.tensor(log_probs).to(DEVICE)
            values_buffer[step] = torch.tensor(values).to(DEVICE)

            # Step all environments
            next_states, step_rewards, terminated, truncated, infos = envs.step(raw_actions)
            
            # Manual episode tracking - update cumulative rewards and lengths
            episode_reward_tracker += step_rewards
            episode_length_tracker += 1
            
            # Check for episode completions
            dones = terminated | truncated
            for env_idx in range(NUM_ENVS):
                if dones[env_idx]:
                    # Episode completed for this environment
                    episode_rewards.append(episode_reward_tracker[env_idx])
                    rollout_episode_rewards.append(episode_reward_tracker[env_idx])
                    # Reset tracker for this environment
                    episode_reward_tracker[env_idx] = 0
                    episode_length_tracker[env_idx] = 0
            
            # Store rewards and dones
            rewards_buffer[step] = torch.tensor(step_rewards).to(DEVICE)
            dones_buffer[step] = torch.tensor(terminated).to(DEVICE)

            
            # Update states
            states = torch.Tensor(next_states).to(DEVICE)
            total_steps += NUM_ENVS
        
        # Bootstrap value for last state
        with torch.no_grad():
            next_values = agent.policy.get_value(states).reshape(NUM_ENVS)
        
        # Flatten buffers for batch processing: (steps, envs, ...) -> (steps*envs, ...)
        obs_flat = obs_buffer.reshape(-1, *envs.single_observation_space.shape)
        actions_flat = actions_buffer.reshape(-1, *envs.single_action_space.shape)
        log_probs_flat = log_probs_buffer.reshape(-1)
        values_flat = values_buffer.reshape(-1)
        
        # Compute returns and advantages using GAE (per environment)
        returns_list = []
        advantages_list = []
        
        for env_idx in range(NUM_ENVS):
            env_rewards = rewards_buffer[:, env_idx]
            env_values = values_buffer[:, env_idx]
            env_dones = dones_buffer[:, env_idx]
            env_next_value = next_values[env_idx]
            
            # Compute GAE for this environment
            env_returns, env_advantages = compute_gae(
                env_rewards, env_values, 
                env_dones[-1].item(), env_dones, 
                env_next_value
            )
            returns_list.append(env_returns)
            advantages_list.append(env_advantages)
        
        # Stack returns and advantages from all environments
        returns_flat = torch.cat(returns_list).to(DEVICE)
        advantages_flat = torch.cat(advantages_list).to(DEVICE)
        
        rollouts = {
            'states': obs_flat,
            'actions': actions_flat,
            'returns': returns_flat,
            'advantages': advantages_flat,
            'values': values_flat,
            'log_probs': log_probs_flat,
        }

        update_metrics = agent.update(rollouts)
        update_count += 1
        
        # Print episode statistics for this rollout
        if len(rollout_episode_rewards) > 0:
            mean_reward = np.mean(rollout_episode_rewards)
            std_reward = np.std(rollout_episode_rewards)
            print(f" Done! | Policy Loss: {update_metrics['policy_loss']:.4f} | Value Loss: {update_metrics['value_loss']:.4f}")
            print(f"  → {len(rollout_episode_rewards)} episodes completed | Mean: {mean_reward:.2f} | Std: {std_reward:.2f}")
            
            # Show overall average if we have enough episodes
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f'  → Avg Reward (last 10 eps): {avg_reward:.2f}')
        else:
            print(f" Done! | Policy Loss: {update_metrics['policy_loss']:.4f} | Value Loss: {update_metrics['value_loss']:.4f}")

      
        log_dict = {
            "policy_loss": update_metrics['policy_loss'],
            "value_function_loss": update_metrics['value_loss'],
            "total_loss": update_metrics['total_loss'],
            "average_episode_reward": avg_reward if len(episode_rewards) >= 10 else 0,
            "total_steps": total_steps,
        }

        if WANDB_AVAILABLE:
            wandb.log(log_dict)
    
    # Cleanup
    envs.close()
    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    env_name = 'CarRacing-v3'
    train(env_name=env_name)