import torch
import torch.nn as nn
import numpy as np
from .base import BaseAgent
from .networks import PolicyNetwork, ValueNetwork
from .utils import compute_gae_lambda, normalize_advantages
import torch.optim as optim
import os

class VPGAgent(BaseAgent):
    def __init__(self, learning_rate=0.0003, vf_lr=0.001, gamma=0.99, lam=0.97, train_v_iters=80, max_ep_len=1000):
        super().__init__(learning_rate, gamma)
        self.vf_lr = vf_lr
        self.lam = lam
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

    def update(self):
        # convert list to tensor on device
        states_tensor = torch.FloatTensor(np.array(self.states)).permute(0, 3, 1, 2).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(self.rewards).to(self.device)
        
        # log_probs are already tensors, just stack
        # if two tensors, x = [1, 2, 3], y = [4, 5, 6], then torch.stack(x, y) = tensor([[1, 2, 3],[4, 5, 6]])
        # c.f torch.cat(x, y) = tensor([1, 2, 3, 4, 5, 6])
        log_probs_tensor = torch.stack(self.log_probs).to(self.device)
        
        values_tensor = torch.stack(self.values).to(self.device)
        dones_tensor = torch.FloatTensor(self.dones).to(self.device)

        # Compute next values for GAE
        # For each state, s_t in the trajectory, compute the value of the next state, V(s_{t+1})
        # Terminal states (dones[i] = True), Last state in the trajectory (i == len(self.states) - 1) will have next value 0 as no future rewards after that point
        next_val_buf = []
        with torch.no_grad():
            for i in range(len(self.states)):
                if self.dones[i]: # terminal state
                    next_val_buf.append(0.0)
                elif i == len(self.states) - 1: # last step in buffer
                    next_val_buf.append(0.0)
                else:
                    next_state = states_tensor[i+1].unsqueeze(0)
                    next_val = self.value_network(next_state).item()
                    next_val_buf.append(next_val) 
        
        next_values = torch.tensor(next_val_buf, device=self.device, dtype=torch.float32)
        
        # Calculate advantages
        advantages_tensor, returns_tensor = compute_gae_lambda(
            rewards_tensor,
            values_tensor,
            next_values,
            dones_tensor,
            gamma=self.gamma,
            lam=self.lam
        )
        
        # Normalize advantages
        advantages_tensor = normalize_advantages(advantages_tensor)

        # Recompute log_probs with gradients enabled (for policy update)
        # Ensure inputs are on device
        log_probs_tensor_grad = self.policy_network.get_log_prob(states_tensor, actions_tensor)

        # Policy update: maximize E[log π(a|s) * A]
        # Loss = -mean(log π(a|s) * A)
        policy_loss = -(log_probs_tensor_grad * advantages_tensor).mean()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        # Value function update
        vf_loss_avg = 0
        for _ in range(self.train_v_iters):
            # Recompute values with gradients
            v_pred = self.value_network(states_tensor)
            # MSE loss: (V(s) - G)^2
            vf_loss = ((v_pred - returns_tensor) ** 2).mean()
            
            self.value_optimizer.zero_grad()
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
            self.value_optimizer.step()

            vf_loss_avg += vf_loss.item()

        vf_loss_avg /= self.train_v_iters

        self.clear_memory()

        return policy_loss.item(), vf_loss_avg, None
