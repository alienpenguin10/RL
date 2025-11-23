"""
Utility functions for VPG: GAE-Lambda computation and other helpers
"""
from math import e
import numpy as np
import torch


def compute_gae_lambda(rewards, values, next_values, dones, gamma=0.99, lam=0.97):
    """
    Compute Generalized Advantage Estimation (GAE-Lambda)
    
    Args:
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
        values: List of value estimates [V(s_0), V(s_1), ..., V(s_{T-1})]
        next_values: List of next state values [V(s_1), V(s_2), ..., V(s_T)]
                     For terminal states, next_value should be 0
        dones: List of done flags [done_0, done_1, ..., done_{T-1}]
        gamma: Discount factor
        lam: GAE-Lambda parameter (trade-off between bias and variance)
    
    Returns:
        advantages: Tensor of advantages [A_0, A_1, ..., A_{T-1}]
        returns: Tensor of returns-to-go [G_0, G_1, ..., G_{T-1}]
    """
    # Convert to tensors if needed
    if isinstance(rewards, list):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    if isinstance(values, list):
        values = torch.tensor(values, dtype=torch.float32)
    if isinstance(next_values, list):
        next_values = torch.tensor(next_values, dtype=torch.float32)
    if isinstance(dones, list):
        dones = torch.tensor(dones, dtype=torch.float32)
    
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0
    
    # Compute GAE backwards from the end of the trajectory
    for t in reversed(range(T)):
        if dones[t]:
            # Terminal state: next value is 0
            delta = rewards[t] - values[t]
            gae = delta
        else:
            # Non-terminal: use next value
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
        
        advantages[t] = gae
    
    # Compute returns = advantages + values
    returns = advantages + values
    
    return advantages, returns


def compute_returns_simple(rewards, discount_factor=0.99):
    """
    Simple returns computation (backward pass)
    G_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
    """
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = discount_factor * G + reward
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sum
    Used for computing returns efficiently
    """
    x = np.asarray(x, dtype=np.float32)
    return torch.tensor(np.array([sum([discount**i * x[t+i] for i in range(len(x)-t)]) 
                                   for t in range(len(x))]), dtype=torch.float32)


def normalize_advantages(advantages):
    """
    Normalize advantages to have mean 0 and std 1
    """
    if isinstance(advantages, torch.Tensor):
        mean = advantages.mean()
        std = advantages.std()
        return (advantages - mean) / (std + 1e-8)
    else:
        raise ValueError("Advantages must be a torch.Tensor")
    

def compute_returns(rewards, discount_factor=0.99):
    #Compute return G_t = Σ(k=t to T-1) γ^(k-t) * r_k
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = discount_factor * G + reward
        returns.insert(0, G) # Insert at the beginning of the list so the first return will be at the last index by the end
    return torch.tensor(returns)


def compute_advantages(returns, values, discount_factor=0.99):
    # Advantage A(s,a) = G_t - V(s)
    advantages = returns - values
    return advantages
