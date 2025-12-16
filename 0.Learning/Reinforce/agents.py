"""
Imports the networks, creates instances, owns optimizers, and implements the learning algorithm. It has methods like select_action(), compute_returns(), update().
"""

import numpy as np
import torch
from networks import PolicyNetwork, ValueNetwork
from utils import compute_returns, compute_advantages, normalize_advantages
import torch.optim as optim

class REINFORCEAgent:
    def __init__(self, learning_rate=0.001, discount_factor=0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Initialize networks
        self.policy_network = PolicyNetwork()
        self.value_network = ValueNetwork()
        # Optimizers
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        # Storage for episode data
        self.reset_episode()
    
    def reset_episode(self):
        # Resets the episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def store_transition(self, state, action, reward):
        # Optionally stores a transition (if using a buffer)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def select_action(self, state):
        # Select action for continous action space
        # Returns action array and stores log prob for training.

        # Convert state to tensor and add batch dimension
        # State comes as (96, 96, 3), need (1, 3, 96, 96) for CNN
        state_tensor = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0)

        # Get action and log_prob from policy network
        # log_prob will have gradients through policy parameters (needed for backprop)
        action, log_prob = self.policy_network.select_action(state_tensor)
        
        # Store log_prob for training (has gradients enabled)
        # log_prob has shape (1, 3), sum across action dimensions to get scalar
        self.log_probs.append(log_prob.sum(dim=1).squeeze(0))
        
        # Also compute and store value estimate
        with torch.no_grad():
            value = self.value_network(state_tensor)
        self.values.append(value.squeeze(0))

        # Return action as numpy array for environment
        return action.squeeze(0).cpu().numpy()

    def update(self):
        # Performs one learning update given collected experience
        # Called at the end of each episode to learn from collected experience.

        # Step 1: Convert raw rewards to returns using discount factor
        # [r1, r2, r3, ...] -> [r1 + γr2 + γ^2r3 + ..., r2 + γr3 + γ^2r4 + ..., ...]
        returns = compute_returns(self.rewards, self.discount_factor)
        
        # Step 2: Timesteps are stored as lists, need to convert to tensors for training.
        # States in self.states are 3D images with shape: (T, 96, 96, 3) - T images, each 96×96 pixels, 3 color channels
        # States tensor needs to be (T, 3, 96, 96) for pytorch CNN.
        states_tensor = torch.FloatTensor(self.states).permute(0, 3, 1, 2)

        # Actions are stored in self.actions as numpy arrays with shape: (T, 3) - T actions, each with 3 components (steering, gas, brake)
        actions_tensor = torch.FloatTensor(self.actions)

        # Returns are already a tensor with shape: (T,) - T returns
        returns_tensor = returns

        # Step 3: Use stored log_probs (computed with gradients during select_action)
        # Log probs are stored in self.log_probs as a list of tensors with shape: (T,) - T log probabilities
        log_probs_tensor = torch.stack(self.log_probs)
        
        # Step 4: Recompute values with gradients enabled for value network training
        # Compute value estimates for all states (with gradients enabled)
        values_tensor = self.value_network(states_tensor).squeeze()

        # Step 5: Compute advantages using value network
        # RECALL: Advantage A(s,a) = G_t - V(s)
        # Tells us: how much better is this action compared to what we expected?
        with torch.no_grad():
            # Actual return" minus "what we predicted"
            # Shape: (T,) - one advantage per timestep
            advantages = compute_advantages(returns_tensor, values_tensor)

            # Normalize advantages to have mean 0 and std 1
            advantages = normalize_advantages(advantages)


        # Step 6: Update policy network using REINFORCE gradient
        # We want to make actions with positive advantages more likely, and actions with negative advantages less likely.

        # REINCFORCE loss: ∇_θ J(θ) ≈ G_t * ∇_θ log π_θ(a_t | s_t) vs L = -log π(aₜ|sₜ) × A(sₜ,aₜ)
        # Negative sign - because we want to maximize the reward (gradient ascent)
        # π(aₜ|sₜ) - probability of action aₜ given state sₜ
        # A(sₜ,aₜ) - advantage of action aₜ given state sₜ
        # But optimizers do gradient Descent, so negate the objective
        # Option 1: Use returns directly
        #   Gradient: ∇θ log π(a|s) × G(s)
        #   Problem: High variance (G varies wildly between episodes)
        
        # Option 2: Use advantages (baseline subtraction)
        #   Gradient: ∇θ log π(a|s) × [G(s) - V(s)]
        #   Benefit: Lower variance, faster learning


        # policy_loss ← -MEAN(log_probs_tensor × advantages)
        # If advantage > 0: action was good, increase log_prob → increase probability
        # If advantage < 0: action was bad, decrease log_prob → decrease probability
        policy_loss = -torch.mean(log_probs_tensor * advantages)

        # Backpropagation through policy network
        # ZERO_GRADIENTS(policy_optimizer)
        self.optimizer.zero_grad()      
        # BACKWARD(policy_loss)                  // Compute ∇θ L with respect to policy params
        policy_loss.backward()
        # CLIP_GRADIENTS(policy_network, 0.5)   // Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        # STEP(policy_optimizer)                // θ ← θ - α·∇θ L
        self.optimizer.step()


        # Step 7: Make value predictions V(s) closer to actual returns G(s)
        # Mean Squared Error loss: L = (V(s) - G(s))²
        value_loss = torch.mean((values_tensor - returns_tensor) ** 2)
        # Backpropagation through value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # CLIP_GRADIENTS(value_network, 0.5)   // Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
        # STEP(value_optimizer)                // θ ← θ - α·∇θ L
        self.value_optimizer.step()

        # Step 8: Reset episode data for next episode
        self.reset_episode()

        return policy_loss.item(), value_loss.item()
    
    def save_model(self, filepath):
        """Save policy and value network models"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)