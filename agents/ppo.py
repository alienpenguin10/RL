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
import torch.optim as optim
import os

class PPOAgent(BaseAgent):
    def __init__(self, learning_rate=0.0003, vf_lr=0.001, gamma=0.99, lam=0.97, clip_ratio=0.2, target_k1=0.01, train_pi_iters=10, train_v_iters=40, max_ep_len=1000):
        super().__init__(learning_rate, gamma)
        self.vf_lr = vf_lr
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_k1 = target_k1
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.max_ep_len = max_ep_len

        # Initialize networks and move to device
        self.policy_network = PolicyNetwork().to(self.device)
        self.value_network = ValueNetwork().to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=vf_lr)

    def select_action(self, state):
        state_tensor = self.preprocess_state(state)
        action, log_prob = self.policy_network.step(state_tensor)

        with torch.no_grad():
            value = self.value_network(state_tensor)

        return action.squeeze(0).cpu().numpy(), log_prob, value.squeeze(0)

    def compute_loss_pi(self, data):
        states, actions, advantages, log_probs_old = data['states'], data['actions'], data['advantages'], data['log_probs_old']
        
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
        loss_pi = -torch.min(surr1, surr2).mean()
        
        # # Useful extra info (Approximate KL Divergence for early stopping)
        # http://joschu.net/blog/kl-approx.html
        approx_kl = (log_probs - log_probs_old).pow(2).mean()

        return loss_pi, approx_kl

    def update(self):
        # convert list to tensor on Device
        # states_tensor = torch.FloatTensor(np.array(self.states)).permute(0, 3, 1, 2).to(self.device)
        # Stack the list of tensors directly
        # squeeze(1) is needed because preprocess adds a batch dim of 1
        states_tensor = torch.cat(self.states).squeeze(1).to(self.device)
        actions_tensor = torch.FloatTensor(self.actions).to(self.device)
        log_probs_old_tensor = torch.stack(self.log_probs).detach().to(self.device)
        rewards_tensor = torch.FloatTensor(self.rewards).to(self.device)
        values_tensor = torch.stack(self.values).to(self.device)
        dones_tensor = torch.FloatTensor(self.dones).to(self.device)

        # 2. Compute GAE (Generalized Advantage Estimation)
        # We need the value of the NEXT state to calculate advantages accurately
        next_val_buf = []
        with torch.no_grad():
            for i in range(len(self.states)):
                if self.dones[i]:
                    next_val_buf.append(0.0)
                elif i == len(self.states) - 1:
                    next_val_buf.append(0.0)  # Assumption: End of batch is terminal or handled elsewhere
                else:
                    next_state = states_tensor[i+1].unsqueeze(0)
                    next_val = self.value_network(next_state).item()
                    next_val_buf.append(next_val)
        
        next_values = torch.tensor(next_val_buf, device=self.device, dtype=torch.float32)
        
        # Calculate advantages
        advantages_tensor, returns_tensor = compute_gae_lambda(rewards_tensor, values_tensor, next_values, dones_tensor, gamma=self.gamma, lam=self.lam)
        
        # Normalize advantages
        advantages_tensor = normalize_advantages(advantages_tensor).detach()
        
        data = {
                'states': states_tensor,
                'actions': actions_tensor,
                'advantages': advantages_tensor,
                'log_probs_old': log_probs_old_tensor
            }

        # 3. Policy Update Loop (with Early Stopping)
        for i in range(self.train_pi_iters):
            self.policy_optimizer.zero_grad()
            loss_pi, approx_kl =  self.compute_loss_pi(data)
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
            self.policy_optimizer.step()

        # 4. Value Function Update Loop
        # (Same as VPG, but usually more iterations in PPO)
        vf_loss_avg = 0
        for _ in range(self.train_v_iters):
            self.value_optimizer.zero_grad()
            v_pred = self.value_network(states_tensor)
            vf_loss = ((v_pred - returns_tensor) ** 2).mean()
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
            self.value_optimizer.step()

            vf_loss_avg += vf_loss.item()

        vf_loss_avg /= self.train_v_iters

        self.clear_memory()
        return loss_pi.item(), vf_loss_avg/ self.train_v_iters, {"kl": approx_kl}
        