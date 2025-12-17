"""
PPO Agent implementing Proximal Policy Optimization
Key differences from REINFORCE:
- Uses a surrogate objective to ensure monotonic policy improvement
- Uses a clipping operation to prevent large policy updates
- Uses a value function to estimate the advantage
- Uses a KL divergence penalty to ensure the policy is not too different from the previous policy
"""
import torch
import torch.nn as nn
import numpy as np
from .base import BaseAgent
from .networks import PolicyNetwork, ValueNetwork
from .utils import compute_gae_lambda, normalize_advantages
from .rollout_buffer import RolloutBuffer
import torch.optim as optim
import os

class PPOAgent(BaseAgent):
    def __init__(self, state_dim=(3, 96, 96), action_dim=3, learning_rate=0.0003, vf_lr=0.001, gamma=0.99, lam=0.97, clip_ratio=0.2, target_k1=0.01, entropy_coef=0.01, entropy_decay=1, mini_batch_size=128, train_pi_iters=10, train_v_iters=40, buffer_size=2048):
        super().__init__(learning_rate, gamma)
        self.vf_lr = vf_lr
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_k1 = target_k1
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.mini_batch_size = mini_batch_size
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.buffer_size = buffer_size
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(buffer_size=buffer_size, state_dim=state_dim, action_dim=action_dim, device=self.device, gamma=gamma, lam=lam)

        # Initialize networks and move to device
        self.policy_network = PolicyNetwork(num_frames=state_dim[0]).to(self.device)
        self.value_network = ValueNetwork(num_frames=state_dim[0]).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=vf_lr)

    def select_action(self, state):
        state_tensor = self.preprocess_state(state)
        with torch.no_grad():
            action, log_prob = self.policy_network.step(state_tensor)
            value = self.value_network(state_tensor)

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.squeeze(0)

    def store_transition(self, state, action, reward, log_prob, done, value=None):
        state_tensor = self.preprocess_state(state).squeeze(0)
        self.rollout_buffer.store(state_tensor, action, log_prob, reward, done, value)

    def finish_path(self, last_val=0):
        """Wrapper to finish the path in the buffer"""
        self.rollout_buffer.finish_path(last_val)

    def compute_loss_pi(self, states, actions, advantages, log_probs_old, training_step=None):
        
        # 1. Get current log_probs from the policy for the states and actions we saw
        # We need to manually recreate the distribution logic here or add a method to PolicyNetwork
        # reusing get_log_prob logic:
        means, log_stds = self.policy_network(states)
        stds  = torch.exp(log_stds)
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(actions).sum(dim=1) 

        # 2. Calculate ratio (pi_theta(a|s) / pi_theta_old(a|s))
        # log(a/b) = log(a) - log(b) => a/b = exp(log(a) - log(b))
        ratio = torch.exp(log_probs - log_probs_old)

        # 3. Calculate Surrogate Objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

        # 4. Calculate PPO Loss (Maximize objective -> Minimize negative objective)
        entropy = dist.entropy().sum(dim=1).mean()
        if self.entropy_decay < 1 and training_step is not None:
            # Decay entropy coefficient over training steps
            entropy_coef = self.entropy_coef * (self.entropy_decay ** training_step)
        else:
            entropy_coef = self.entropy_coef
        
        loss_pi = -(torch.min(surr1, surr2).mean() + entropy_coef * entropy)
        
        # # Useful extra info (Approximate KL Divergence for early stopping)
        # http://joschu.net/blog/kl-approx.html
        # approx_kl = (log_probs - log_probs_old).pow(2).mean()
        approx_kl = -1

        return loss_pi, approx_kl, entropy

    def update(self, training_step=None):
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
        
                loss_pi, approx_kl, entropy = self.compute_loss_pi(batch_states, batch_actions, batch_adv, batch_log_probs_old, training_step)
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                
                # if approx_kl > 1.5 * self.target_k1:
                #     print(f"Early stopping at iteration {_} due to KL={approx_kl:.4f}")


        # Value Update
        for _ in range(self.train_v_iters):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_returns = returns[idx]
                
                self.value_optimizer.zero_grad()
                v_pred = self.value_network(batch_states)
                vf_loss = ((v_pred - batch_returns) ** 2).mean()
                vf_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.value_optimizer.step()

        self.clear_memory()
        return loss_pi.item(), vf_loss.item()