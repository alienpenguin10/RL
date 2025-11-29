from .base import BaseAgent # Assuming modular structure
from .networks import PolicyNetwork
from .utils import compute_returns, normalize_advantages
import torch
import torch.optim as optim

class REINFORCEAgent(BaseAgent):
    def __init__(self, learning_rate=0.001, gamma=0.99):
        super().__init__(learning_rate, gamma)
        self.policy_network = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
    
    def select_action(self, state):
        state_tensor = self.preprocess_state(state)
        
        action, log_prob = self.policy_network.step(state_tensor)

        # Return action as numpy (thats what env expects) also remove batch dimension, keep log_prob as tensor for graph
        # action:(batch_size, 3) = [[steering, gas, brake]] -> action.squeeze(0):(3,) = [steering, gas, brake]
        return action.squeeze(0).cpu().numpy(), log_prob

    def update(self):
        # Calculate returns (G_t)
        returns = compute_returns(self.rewards, self.gamma)
        returns = normalize_advantages(returns).to(self.device) # Normalize & GPU
        
        # Stack log_probs (they are already tensors attached to graph)
        log_probs_tensor = torch.cat(self.log_probs).to(self.device)

        # REINFORCE Loss: -mean( log_prob * G_t )
        # Gradient ascent via descent
        # RL goal = maximize expected return, J(θ) = E[∑ R_t | π_θ]
        # Policy Gradient Theorem: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
        # Prob: Neural network optimizers (SGD, Adam) minimize loss, not maximize rewards
        # To maximize J(θ) = minimize -J(θ)
        policy_loss = -(log_probs_tensor * returns).mean()

        self.optimizer.zero_grad()
        policy_loss.backward() # How much each policy parameter contributes to the loss
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        self.optimizer.step() # Update the policy parameters by new_weight = current_weight - learning_rate * gradient
        
        self.clear_memory() # Use standardized name
        return policy_loss.item()
