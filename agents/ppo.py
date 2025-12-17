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
    def __init__(self, learning_rate=0.0003, vf_lr=0.001, gamma=0.99, lam=0.95, clip_ratio=0.2, target_k1=0.01, train_pi_iters=10, train_v_iters=10, max_ep_len=1000, entropy_coeff=0.01):
        super().__init__(learning_rate, gamma)
        self.vf_lr = vf_lr
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_k1 = target_k1
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.max_ep_len = max_ep_len
        self.entropy_coeff = entropy_coeff

        # Initialize networks and move to device
        self.policy_network = PolicyNetwork().to(self.device)
        self.value_network = ValueNetwork().to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=vf_lr)

    def select_action(self, state):
        state_tensor = self.preprocess_state(state)
        # FIX: Use no_grad() for the entire block. 
        # This disables gradient tracking for inference, which fixes the .numpy() error
        # and prevents memory leaks during rollouts.
        with torch.no_grad():
            action_squashed, action_raw, log_prob = self.policy_network.get_action_and_log_prob(state_tensor)
            value = self.value_network(state_tensor)

        return action_raw.squeeze(0).cpu().numpy(), log_prob, value.squeeze(0), action_squashed.squeeze(0).cpu().numpy()

    def compute_loss_pi(self, data):
        states, actions_raw, advantages, log_probs_old = data['states'], data['actions'], data['advantages'], data['log_probs_old']
        
        # 1. Get current log_probs from the policy for the states and actions we saw
        # We need to manually recreate the distribution logic here or add a method to PolicyNetwork
        # reusing get_log_prob logic:
        # means, log_stds = self.policy_network(states)
        # stds  = torch.exp(log_stds)
        # dist = torch.distributions.Normal(means, stds)
        # log_probs = dist.log_prob(actions).sum(dim=1) 

        # Pass RAW actions to the network to recalculate probability
        _, _, log_probs = self.policy_network.get_action_and_log_prob(states, action=actions_raw)

        # 2. Calculate ratio (pi_theta(a|s) / pi_theta_old(a|s))
        # log(a/b) = log(a) - log(b) => a/b = exp(log(a) - log(b))
        ratio = torch.exp(log_probs - log_probs_old)

        # 3. Calculate Surrogate Objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

        # 4. Calculate PPO Loss (Maximize objective -> Minimize negative objective)
        loss_pi = -torch.min(surr1, surr2).mean()

        # 5. Add Entropy Bonus
        #entropy_loss = -self.entropy_coeff * dist.entropy().mean()
        #loss_pi += entropy_loss
        
        # # Useful extra info (Approximate KL Divergence for early stopping)
        # http://joschu.net/blog/kl-approx.html
        #approx_kl = (log_probs - log_probs_old).pow(2).mean()

        approx_entropy = -log_probs.mean() 
        loss_pi -= self.entropy_coeff * approx_entropy
        
        approx_kl = (log_probs_old - log_probs).mean().item() # http://joschu.net/blog/kl-approx.html formulation

        return loss_pi, approx_kl

    def update(self, last_state, last_done):
        # 1. Prepare Tensors (Same as before)
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        # Was: torch.stack(self.log_probs).detach().to(self.device)
        # Fix: Add .flatten() or .squeeze() to ensure shape is [Batch_Size], not [Batch_Size, 1]
        log_probs_old_tensor = torch.stack(self.log_probs).detach().to(self.device).flatten()
        rewards_tensor = torch.FloatTensor(self.rewards).to(self.device)
        values_tensor = torch.stack(self.values).to(self.device)
        
        # 2. Bootstrapping: Calculate value of the very last state
        with torch.no_grad():
            if last_done:
                last_val = 0.0
            else:
                # We need to preprocess the last state just like the others
                last_state_t = self.preprocess_state(last_state)
                last_val = self.value_network(last_state_t).item()

        # 3. Compute GAE with Correct Bootstrapping
        # We rewrite this loop to be explicit and correct
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae_lam = 0
        
        # Convert values to numpy for faster CPU processing in this loop
        values_np = values_tensor.cpu().numpy().flatten()
        rewards_np = rewards_tensor.cpu().numpy()
        dones_np = np.array(self.dones, dtype=np.float32)
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_val
            else:
                next_non_terminal = 1.0 - dones_np[t]
                next_value = values_np[t + 1]
            
            delta = rewards_np[t] + self.gamma * next_value * next_non_terminal - values_np[t]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        advantages_tensor = torch.tensor(advantages, device=self.device)
        # Target for Value Function: V_target = Advantage + V_old
        returns_tensor = advantages_tensor + values_tensor
        
        # Normalize Advantages (Critical for PPO stability)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # 4. Minibatch Training Loop (The RLlib way)
        dataset_size = len(states_tensor)
        indices = np.arange(dataset_size)
        minibatch_size = 128  # RLlib default is usually 128 or 256
        
        avg_pi_loss = 0
        avg_vf_loss = 0
        avg_kl = 0
        updates = 0

        # Create a dict for the data to slice easily
        # Note: We detach everything to ensure we don't backprop into data collection
        data = {
            'states': states_tensor,
            'actions': actions_tensor,
            'advantages': advantages_tensor,
            'log_probs_old': log_probs_old_tensor,
            'returns': returns_tensor
        }

        target_kl = 0.015
        for i in range(self.train_pi_iters):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]
                
                # Slice Minibatch
                mb_data = {k: v[idx] for k, v in data.items()}
                
                # Update Policy
                self.policy_optimizer.zero_grad()
                loss_pi, approx_kl = self.compute_loss_pi(mb_data)
                # KL EARLY STOPPING
                if approx_kl > 1.5 * target_kl:
                    print(f"Early stopping at epoch {i} due to KL: {approx_kl:.4f}")
                    break # Break minibatch loop
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Update Value Function
                # (You can do this in a separate loop, but doing it here is fine/standard)
                self.value_optimizer.zero_grad()
                v_pred = self.value_network(mb_data['states'])
                # Optional: Clip Value loss like RLlib does, but MSE is okay for now
                loss_vf = ((v_pred - mb_data['returns']) ** 2).mean()
                loss_vf.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.value_optimizer.step()

                # Logging
                avg_pi_loss += loss_pi.item()
                avg_vf_loss += loss_vf.item()
                avg_kl += approx_kl
                updates += 1
            
            # If we broke inner loop, break outer loop too
            if approx_kl > 1.5 * target_kl:
                break

        self.clear_memory()
        
        return avg_pi_loss / updates, avg_vf_loss / updates, {"kl": avg_kl / updates}
