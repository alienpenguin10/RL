import torch
import torch.nn as nn
from agents.base import BaseAgent
from agents.networks import PolicyNetwork, QNetwork
from agents.replay_buffer import ReplayBuffer
import pickle
import numpy as np
import torch.nn.functional as F


class SACAgent(BaseAgent):
    def __init__(self, obs_dim, action_dim, batch_size=256,
                 policy_lr=3e-4, q_lr=3e-4, policy_weight_decay=1e-4,
                 q_weight_decay=1e-4, alpha_lr=3e-4, alpha_weight_decay=1e-4,
                 gamma=0.99, tau=0.005, alpha=0.1, buffer_capacity=1000000,
                 q_hidden_dims=[256, 256], policy_hidden_dims=[256, 256]):
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        super().__init__(learning_rate=policy_lr, gamma=gamma)

        self.tau = tau
        self.alpha = alpha

        self.batch_size = batch_size

        # entropy target based on action_dim (CarRacing: action_dim = 3 -> target_entropy = -3)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(self.alpha))),
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr, weight_decay=alpha_weight_decay)

        self.policy_network = PolicyNetwork(obs_dim, action_dim, hidden_dims=policy_hidden_dims).to(self.device)
        self.policy_optimiser = torch.optim.Adam(
            self.policy_network.parameters(), 
            lr=policy_lr,
            weight_decay=policy_weight_decay
        )

        # 2 CRITICS
        self.q_net_1 = QNetwork(obs_dim, action_dim, hidden_dims=q_hidden_dims).to(self.device)
        self.q_net_1_target = QNetwork(obs_dim, action_dim, hidden_dims=q_hidden_dims).to(self.device)
        self.q_optim_1 = torch.optim.Adam(
            self.q_net_1.parameters(), 
            lr=q_lr,
            weight_decay=q_weight_decay
        )

        self.q_net_2 = QNetwork(obs_dim, action_dim, hidden_dims=q_hidden_dims).to(self.device)
        self.q_net_2_target = QNetwork(obs_dim, action_dim, hidden_dims=q_hidden_dims).to(self.device)
        self.q_optim_2 = torch.optim.Adam(
            self.q_net_2.parameters(), 
            lr=q_lr,
            weight_decay=q_weight_decay
        )
        
        # Initialize target networks
        self.q_net_1_target.load_state_dict(self.q_net_1.state_dict())
        self.q_net_2_target.load_state_dict(self.q_net_2.state_dict())
        
        # Freeze target network parameters (they're updated via soft update)
        for param in self.q_net_1_target.parameters():
            param.requires_grad = False
        for param in self.q_net_2_target.parameters():
            param.requires_grad = False

    
    def save_replay_buffer(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load_replay_buffer(self, filepath):
        with open(filepath, "rb") as f:
            self.replay_buffer = pickle.load(f)
            print("Replay buffer size after loading:", len(self.replay_buffer))
    
    
    def clear_memory(self):
        """Resets the replay buffer by creating a new instance, discarding old transitions"""
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer.capacity)

    def select_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy_network.step(state_tensor, deterministic)
        return action.cpu().numpy()[0].astype(np.float32)

    def store_transition(self, state, action, reward, next_state, log_prob, done, value=None):
        """
        Store Experience in replay buffer
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self.device)

        # === Q-NETWORK UPDATE ===
        # Compute target for Q functions
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy_network.step(batch['next_obs'])
            qf1_next = self.q_net_1_target(batch['next_obs'], next_action)
            qf2_next = self.q_net_2_target(batch['next_obs'], next_action)
            min_q_next = torch.min(qf1_next, qf2_next)
            y = batch['rewards'] + (1 - batch['dones']) * self.gamma * (min_q_next - self.alpha * next_log_prob)
        
        # Compute loss for Q functions
        qf_1 = self.q_net_1(batch['obs'], batch['actions'])
        qf_2 = self.q_net_2(batch['obs'], batch['actions'])
        qf1_loss = F.mse_loss(qf_1, y)
        qf2_loss = F.mse_loss(qf_2, y)
        qf_loss = qf1_loss + qf2_loss
        
        # Update Q functions
        self.q_optim_1.zero_grad()
        self.q_optim_2.zero_grad()
        qf_loss.backward()
        self.q_optim_1.step()
        self.q_optim_2.step()
        
        # === POLICY UPDATE ===
        acts, log_prob, _ = self.policy_network.step(batch['obs'])
        qf1 = self.q_net_1(batch['obs'], acts)
        qf2 = self.q_net_2(batch['obs'], acts)
        min_qf = torch.min(qf1, qf2)
        actor_loss = (self.alpha * log_prob - min_qf).mean()
        
        # Update policy
        self.policy_optimiser.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        self.policy_optimiser.step()
        
        # === ALPHA (Temperature) UPDATE ===
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        # === UPDATE TARGET Q-NETWORKS ===
        for param, target_param in zip(self.q_net_1.parameters(), self.q_net_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q_net_2.parameters(), self.q_net_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Return metrics
        return {
            'critic_loss': qf_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'entropy': -log_prob.mean().item(),
            'q1_mean': qf_1.mean().item(),
            'q2_mean': qf_2.mean().item(),
            'q_optim_1_lr': self.q_optim_1.param_groups[0]['lr'],
            'q_optim_2_lr': self.q_optim_2.param_groups[0]['lr'],
            'policy_optim_lr': self.policy_optimiser.param_groups[0]['lr'],
            'alpha_optim_lr': self.alpha_optim.param_groups[0]['lr'],
        }

    def _soft_update_networks(self, source, target):
        """Polyak averaging for target network updates"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, filepath):
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'q_network_1_state_dict': self.q_net_1.state_dict(),
            'q_network_1_target_state_dict': self.q_net_1_target.state_dict(),
            'q_network_2_state_dict': self.q_net_2.state_dict(),
            'q_network_2_target_state_dict': self.q_net_2_target.state_dict(),
            'policy_optimiser_state_dict': self.policy_optimiser.state_dict(),
            'q_optimiser_1_state_dict': self.q_optim_1.state_dict(),
            'q_optimiser_2_state_dict': self.q_optim_2.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.q_net_1.load_state_dict(checkpoint['q_network_1_state_dict'])
        self.q_net_1_target.load_state_dict(checkpoint['q_network_1_target_state_dict'])
        self.q_net_2.load_state_dict(checkpoint['q_network_2_state_dict'])
        self.q_net_2_target.load_state_dict(checkpoint['q_network_2_target_state_dict'])
        self.policy_optimiser.load_state_dict(checkpoint['policy_optimiser_state_dict'])
        self.q_optim_1.load_state_dict(checkpoint['q_optimiser_1_state_dict'])
        self.q_optim_2.load_state_dict(checkpoint['q_optimiser_2_state_dict'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = checkpoint['alpha']
