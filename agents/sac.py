import torch

from agents.base import BaseAgent
from agents.networks import PolicyNetwork, QNetwork
from agents.replay_buffer import ReplayBuffer
import pickle


class SACAgent(BaseAgent):
    # class level - shared by all instances (add to __init__ for per-agent buffers)
    # replay_buffer = ReplayBuffer(capacity=10000)

    def __init__(self, action_dim, batch_size=256, warmup_factor=4.0,
             policy_lr=3e-4, q_lr=3e-4, policy_weight_decay=1e-4,
             q_weight_decay=1e-4, gamma=0.99, tau=0.05, alpha=0.1,
             actor_update_frequency=2):
        super().__init__(learning_rate=policy_lr, gamma=gamma)

        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.tau = tau
        self.alpha = alpha

        self.batch_size = batch_size
        self.warmup_factor = warmup_factor

        self.actor_update_frequency = actor_update_frequency
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
        self.policy_optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=policy_lr,
                                                 weight_decay=policy_weight_decay)

        # 2 CRITICS
        self.q_net_1 = QNetwork().to(self.device)
        self.q_net_1_target = QNetwork().to(self.device)
        self.q_optim_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=q_lr,
                                          weight_decay=q_weight_decay)

        self.q_net_2 = QNetwork().to(self.device)
        self.q_net_2_target = QNetwork().to(self.device)
        self.q_optim_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=q_lr,
                                          weight_decay=q_weight_decay)
        
        self.q_net_1_target.load_state_dict(self.q_net_1.state_dict())
        self.q_net_2_target.load_state_dict(self.q_net_2.state_dict())

    
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
        action, log_prob = self.policy_network.step_new(state_tensor)   # Sample action and get log-prob 
        return action.squeeze(0).cpu().detach().numpy(), log_prob.detach()

    def store_transition(self, state, action, reward, next_state, log_prob, done, value=None):
        """Store Experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        # If not enough samples in buffer, skip update
        if len(self.replay_buffer) < self.batch_size * self.warmup_factor:
            return None

        self.update_counter += 1
        alpha = self.log_alpha.exp()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-value update (always)
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.step_new(next_states)
            next_log_probs = next_log_probs.unsqueeze(1)
            q1_next = self.q_net_1_target(next_states, next_actions)
            q2_next = self.q_net_2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q_1_loss = ((self.q_net_1(states, actions) - q_target) ** 2).mean()
        q_2_loss = ((self.q_net_2(states, actions) - q_target) ** 2).mean()

        self.q_optim_1.zero_grad()
        q_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net_1.parameters(), max_norm=1.0)
        self.q_optim_1.step()

        self.q_optim_2.zero_grad()
        q_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net_2.parameters(), max_norm=1.0)
        self.q_optim_2.step()

        # Policy and Alpha Update (less frequent)
        if self.update_counter % self.actor_update_frequency == 0:
            # Freeze critic parameters
            for param in self.q_net_1.parameters():
                param.requires_grad = False
            for param in self.q_net_2.parameters():
                param.requires_grad = False

            # Policy Update
            new_actions, log_probs = self.policy_network.step_new(states)
            log_probs = log_probs.unsqueeze(1)
            q1_new = self.q_net_1(states, new_actions)
            q2_new = self.q_net_2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            policy_loss = (alpha * log_probs - q_new).mean()
            self.policy_optimiser.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
            self.policy_optimiser.step()

            # Alpha update
            with torch.no_grad():
                entropy_term = -log_probs.squeeze(1).detach()

            alpha_loss = (self.log_alpha * (entropy_term - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

            # Unfreeze critic parameters
            for param in self.q_net_1.parameters():
                param.requires_grad = True
            for param in self.q_net_2.parameters():
                param.requires_grad = True
        else:
            # Return previous losses if no update
            policy_loss = torch.tensor(0.0)
            alpha_loss = torch.tensor(0.0)

        # Soft update target networks (every step is fine)
        self._soft_update_networks(self.q_net_1, self.q_net_1_target)
        self._soft_update_networks(self.q_net_2, self.q_net_2_target)

        return policy_loss.item(), (q_1_loss.item() + q_2_loss.item()) / 2, {"alpha": self.alpha, "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss}

    def _soft_update_networks(self, source, target):
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