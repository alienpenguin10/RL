import torch
from agents.base import BaseAgent
from agents.networks import PolicyNetwork, QNetwork
from agents.replay_buffer import ReplayBuffer
import pickle
import numpy as np
import torch.nn.functional as F


class SACAgent(BaseAgent):
    def __init__(self, action_dim, batch_size=256, warmup_factor=1.0,  # FIXED: warmup_factor to 1.0
                 policy_lr=3e-4, q_lr=3e-4, policy_weight_decay=1e-4,
                 q_weight_decay=1e-4, gamma=0.99, tau=0.005, alpha=0.1,  # FIXED: tau to 0.005
                 buffer_capacity=1000000):  # ADDED: buffer_capacity parameter
        super().__init__(learning_rate=policy_lr, gamma=gamma)

        # FIXED: Use buffer_capacity parameter
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.tau = tau
        self.alpha = alpha

        self.batch_size = batch_size
        self.warmup_factor = warmup_factor

        # REMOVED: actor_update_frequency (update every step now)
        self.update_counter = 0  # Track update iterations

        # entropy target based on action_dim (CarRacing: action_dim = 3 -> target_entropy = -3)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(self.alpha))),
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=policy_lr)        

        # ACTOR
        self.policy_network = PolicyNetwork().to(self.device)
        self.policy_optimiser = torch.optim.Adam(
            self.policy_network.parameters(), 
            lr=policy_lr,
            weight_decay=policy_weight_decay
        )

        # 2 CRITICS
        self.q_net_1 = QNetwork().to(self.device)
        self.q_net_1_target = QNetwork().to(self.device)
        self.q_optim_1 = torch.optim.Adam(
            self.q_net_1.parameters(), 
            lr=q_lr,
            weight_decay=q_weight_decay
        )

        self.q_net_2 = QNetwork().to(self.device)
        self.q_net_2_target = QNetwork().to(self.device)
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
        self.replay_buffer = ReplayBuffer(capacity=1000000)

    def select_action(self, state):
        state_tensor = self.preprocess_state(state)
        action, log_prob = self.policy_network.step_new(state_tensor)
        return action.squeeze(0).cpu().detach().numpy(), log_prob.detach()

    def store_transition(self, state, action, reward, next_state, done, value=None):
        """
        Store Experience in replay buffer
        FIXED: Removed log_prob parameter (not needed in SAC replay buffer)
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.update_counter += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (states are already uint8 (4, 84, 84))
        states = torch.FloatTensor(np.array(states)).to(self.device) / 255.0
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) / 255.0
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # === Q-NETWORK UPDATE ===
        # Compute target for Q functions
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.step_new(next_states)
            qf1_next = self.q_net_1_target(next_states, next_actions)
            qf2_next = self.q_net_2_target(next_states, next_actions)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha * next_log_probs
            y = rewards.flatten() + (1 - dones.flatten()) * self.gamma * min_qf_next.view(-1)
        
        # Compute loss for Q functions
        qf_1 = self.q_net_1(states, actions).view(-1)
        qf_2 = self.q_net_2(states, actions).view(-1)
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
        acts, log_prob = self.policy_network.step_new(states)
        qf1 = self.q_net_1(states, acts)
        qf2 = self.q_net_2(states, acts)
        min_qf = torch.min(qf1, qf2).view(-1)
        # Negative sign for maximization
        actor_loss = -(min_qf - self.alpha * log_prob).mean()
        
        # Update policy
        self.policy_optimiser.zero_grad()
        actor_loss.backward()
        self.policy_optimiser.step()
        
        # === ALPHA (Temperature) UPDATE ===
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()
        
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
            'entropy': dist.entropy().mean().item(),
            'q1_mean': qf_1.mean().item(),
            'q2_mean': qf_2.mean().item(),
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