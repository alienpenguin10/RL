import gymnasium as gym
from agents.pendulum_buffer import RolloutBuffer
from agents.pendulum_networks import NNetwork
import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys
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


class PendulumAgent:
    def __init__(self, state_dim, action_dim, device='cpu', gamma=0.99, lam=0.95, entropy_coef=0.01, entropy_decay=1.0, clip_ratio=0.2, pi_lr=0.001, vf_lr=0.001, buffer_size=1000, train_pi_iters=10, train_v_iters=10, mini_batch_size=64):
        self.device = device

        self.policy_network = NNetwork(3, 1).to(device)
        self.value_network = NNetwork(3, 1).to(device)

        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=pi_lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=vf_lr)

        self.rollout_buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim, device=device, gamma=gamma, lam=lam)

        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.clip_ratio = clip_ratio
        # use fixed std
        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5)).to(device)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).squeeze().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean = self.policy_network(state_tensor)
            dist = torch.distributions.Normal(mean, self.std)
            action = torch.clamp(2 * dist.sample(), -2, 2).detach()
            log_prob = dist.log_prob(action).sum(dim=1).detach()
            value = self.value_network(state_tensor)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.rollout_buffer.store(state_tensor, action, log_prob, reward, value, done)

    def finish_path(self, last_value=0):
        self.rollout_buffer.finish_path(last_value)
    
    def compute_pi_loss(self, states, actions, advantages, log_probs_old):
        
        # 1. Get current log_probs from the policy for the states and actions we saw
        # We need to manually recreate the distribution logic here or add a method to PolicyNetwork
        # reusing get_log_prob logic:
        means = self.policy_network(states)
        dist = torch.distributions.Normal(means, self.std)
        log_probs = dist.log_prob(actions)

        # 2. Calculate ratio (pi_theta(a|s) / pi_theta_old(a|s))
        # log(a/b) = log(a) - log(b) => a/b = exp(log(a) - log(b))
        ratio = torch.exp(log_probs - log_probs_old)

        # 3. Calculate Surrogate Objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

        if self.entropy_decay != 1.0:
            # Decay entropy coefficient
            self.entropy_coef *= self.entropy_decay
        
        # 4. Calculate PPO Loss (Maximize objective -> Minimize negative objective)
        entropy = dist.entropy().mean()
        pi_loss = -(torch.min(surr1, surr2)).mean() + self.entropy_coef * entropy

        return pi_loss

    def update(self):
        data = self.rollout_buffer.get()
        states, actions, advantages, log_probs_old, values, returns = data['states'], data['actions'], data['advantages'], data['log_probs'], data['values'], data['returns']

        # Mini-batch updates
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        # Policy Update
        for _ in range(self.train_pi_iters):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_adv = advantages[idx]
                batch_log_probs_old = log_probs_old[idx]
                
                self.policy_optimizer.zero_grad()
                pi_loss = self.compute_pi_loss(batch_states, batch_actions, batch_adv, batch_log_probs_old)
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=4.0)
                self.policy_optimizer.step()

        # Value Update
        for _ in range(self.train_v_iters):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_returns = returns[idx]
                
                self.value_optimizer.zero_grad()
                v_pred = self.value_network(batch_states).squeeze(-1)
                vf_loss = nn.MSELoss()(v_pred, batch_returns)
                vf_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 4.0)
                self.value_optimizer.step()

        return pi_loss.item(), vf_loss.item()
    
    def save_model(self, filepath):
        """
        Saves the model to a file
        """
        save_dict = {}
        if hasattr(self, 'policy_network'):
            save_dict['policy_network'] = self.policy_network.state_dict()
        if hasattr(self, 'value_network'):
            save_dict['value_network'] = self.value_network.state_dict()
        torch.save(save_dict, filepath)

""" Hyperparameters """
env_name = 'Pendulum-v1'
num_training_steps = 500000
batch_size = 100
buffer_size = 2000
pi_lr = 0.0001
vf_lr = 0.0001
entropy_coef = 0.05
entropy_decay = 0.99999
clip_ratio = 0.2

def train(max_train_iters=100, save_checkpoints=True):# Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name=f"ppo_{env_name}",
            config={
                "algorithm": "ppo",
                "environment": env_name,
                "max_episodes": max_train_iters,
                "learning_rate": pi_lr,
            }
        )
    
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PendulumAgent(state_dim, action_dim, device, pi_lr=pi_lr, vf_lr=vf_lr, entropy_coef=entropy_coef, entropy_decay=entropy_decay, clip_ratio=clip_ratio, buffer_size=buffer_size)

    # Track current episode for saving on interrupt
    current_episode = [0]  # Use list to allow modification in nested function
    
    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model from episode {current_episode[0]}...")
        agent.save_model(f"./models/ppo_pendulum_{current_episode[0]}_interrupted.pth")
        print(f"Model saved to ./models/ppo_pendulum_{current_episode[0]}_interrupted.pth")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)

    steps = 0
    max_steps = max_train_iters

    while steps < max_steps:
        state, _ = env.reset()
        episode = 0
        episode_reward = 0
        episode_rewards = []

        for t in range(buffer_size):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

            agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
            steps += 1
            episode_reward += reward

            # Handle episode termination
            if done:
                # Finish path with value=0 since it's terminal
                agent.finish_path(last_value=0)
                episode += 1
                current_episode[0] = episode

                # Log episode reward
                episode_rewards.append(episode_reward)

                # Reset for next episode
                episode_reward = 0
                state, _ = env.reset()
            elif t == buffer_size - 1:
                # Buffer is full but episode not done: Bootstrap!
                # We need the value of the current 'state' (which is actually next_state of the last step)
                _, _, last_val = agent.select_action(state)
                agent.finish_path(last_val=last_val)

        # 2. Update after buffer is full
        loss = agent.update()
        policy_loss, vf_loss = loss

        avg_episode_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0

        log_dict = {
            "policy_loss": policy_loss,
            "value_function_loss": vf_loss,
            "entropy": agent.entropy_coef,
            "average_episode_reward": avg_episode_reward,
        }

        if WANDB_AVAILABLE:
            wandb.log(log_dict)

        if save_checkpoints:
            agent.save_model(f"./models/ppo_pendulum_{steps}_checkpoint.pth")
    
    # Save final model
    print(f"\nTraining complete! Saving final model...")
    agent.save_model(f"./models/ppo_pendulum_{max_train_iters-1}_final.pth")
    
    if WANDB_AVAILABLE:
        wandb.finish()
    
    env.close()

if __name__ == "__main__":
    # Check for WandB API key - fail if not set
    if WANDB_AVAILABLE:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY is not set in the environment variables. Please create a .env file with your WandB API key. See README.md for setup instructions.")
    
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    train(max_train_iters=num_training_steps)
