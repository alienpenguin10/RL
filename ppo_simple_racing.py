import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import gymnasium as gym
from agents.networks import ConvNet, ConvNet_StackedFrames
from env_wrapper import ProcessedFrame, FrameStack, ActionRemapWrapper
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
    def __init__(self, num_frames, output_shape):
        super().__init__()
        # Using Orthogonal Initialization
        self.conv = ConvNet_StackedFrames(num_frames=num_frames)
        conv_size = 4096  # Output size from ConvNet (256 channels * 4 height * 4 width)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(conv_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(conv_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, output_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_shape))
    
    def get_obs_features(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0) # To make sure state has a batch dimension
        if (not FRAME_STACKING):
            state_tensor = state.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        else:
            state_tensor = state # Frame stacking already gives (B, frames, H, W)
        features = self.conv(state_tensor).flatten(start_dim=1)
        return features

    def get_value(self, state):
        conv_features = self.get_obs_features(state)
        return self.critic(conv_features)

    def get_action(self, state):
        # Takes a single state -> samples a new action from policy dist
        conv_features = self.get_obs_features(state)
        mean = self.actor(conv_features)
        action_logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        value = self.critic(conv_features)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()
    
    def evaluate(self, states, actions):
        # takes in batch of states and actions -> doesn't sample evaluates the log prob of specific action under the current policy
        # also returns entropy regularization term
        conv_features = self.get_obs_features(states)
        mean = self.actor(conv_features)
        action_logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(conv_features)
        return log_prob, value, entropy

class PPOAgent:
    def __init__(self, env):
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.policy = ActorCritic(num_frames=NUM_FRAMES, output_shape=self.action_size).to(DEVICE)
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

""" Hyperparameters """
LOG_WANDB = True
TOTAL_TIMESTEPS = 20000
HORIZON = 2048 # One episode is 200 steps for pendulum
NUM_UPDATES = int(TOTAL_TIMESTEPS / HORIZON) # 100000 / 2048 = 244
NUM_EPOCHS = 10
NUM_MINIBATCHES = 32 
BATCH_SIZE = HORIZON // NUM_MINIBATCHES # 2048 // 32 = 64
FRAME_STACKING = True
NUM_FRAMES = 5
SKIP_FRAMES = 0

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILONS = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env_name='CarRacing-v3', log_wandb=False):

    # Initialize WandB if available
    if WANDB_AVAILABLE and log_wandb:
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
    
    env = gym.make(f'{env_name}')
    env = ProcessedFrame(env)
    env = FrameStack(env, num_frames=NUM_FRAMES, skip_frames=SKIP_FRAMES)
    env = ActionRemapWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = PPOAgent(env)

    state, _ = env.reset()
    state = torch.Tensor(state).to(DEVICE)
    terminated = 0
    total_steps = 0
    update_count = 0
    # episode_rewards = []
    
    while total_steps < TOTAL_TIMESTEPS:
        states = torch.zeros((HORIZON, *env.observation_space.shape)).to(DEVICE)
        actions = torch.zeros((HORIZON, env.action_space.shape[0])).to(DEVICE)
        rewards = torch.zeros((HORIZON)).to(DEVICE)
        terminateds = torch.zeros((HORIZON)).to(DEVICE)
        values = torch.zeros((HORIZON)).to(DEVICE)
        log_probs = torch.zeros((HORIZON)).to(DEVICE)

        rollout_rewards = []

        # Rollout
        for step in range(HORIZON):
            states[step] = state
            terminateds[step] = torch.tensor(terminated).to(DEVICE)
            with torch.no_grad():
                raw_action, log_prob, value = agent.policy.get_action(state)
                values[step] = torch.tensor(value).to(DEVICE)       
            actions[step] = torch.tensor(raw_action).to(DEVICE)
            log_probs[step] = torch.tensor(log_prob).to(DEVICE)

            clipped_action = np.clip(raw_action, -2.0, 2.0)

            next_state, reward, terminated, truncated, info = env.step(clipped_action)
            # if "episode" in info:
            #     print(f"Global Step: {total_steps}, Episode Return: {info['episode']['r']}, Length: {info['episode']['l']}")
            #     episode_rewards.append(info['episode']['r'])
            rollout_rewards.append(reward)

            # Crucial: Only treat as 'done' if terminated (failure), not truncated (time limit)
            rewards[step] = torch.tensor(reward).to(DEVICE)
            if terminated or truncated:
                next_state, _ = env.reset()
            state = torch.Tensor(next_state).to(DEVICE)
            total_steps += 1
        
        # Boostrap value if not done
        with torch.no_grad():
            next_value = agent.policy.get_value(state).reshape(-1)

        returns, advantages = compute_gae(rewards, values, terminated, terminateds, next_value)
        
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

        # if len(episode_rewards) >= 10:
        #     avg_reward = np.mean(episode_rewards[-10:])
        #     print(f'Update {update_count}: {avg_reward}')
        avg_reward = np.mean(rollout_rewards)

        log_dict = {
            "policy_loss": update_metrics['policy_loss'],
            "value_function_loss": update_metrics['value_loss'],
            "total_loss": update_metrics['total_loss'],
            "average_episode_reward": avg_reward,
        }

        if WANDB_AVAILABLE and log_wandb:
            wandb.log(log_dict)

if __name__ == '__main__':
    env_name = 'CarRacing-v3'
    log_wandb = LOG_WANDB
    train(env_name=env_name, log_wandb=log_wandb)