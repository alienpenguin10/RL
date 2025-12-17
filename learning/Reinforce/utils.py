"""
contains compute_advantages(), compute_policy_loss(), normalize_advantages(). These are pure functions that any agent can call.
"""
import numpy as np
import torch

def compute_returns(rewards, discount_factor=0.99):
     #Compute return G_t = Σ(k=t to T-1) γ^(k-t) * r_k
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = discount_factor * G + reward
        returns.insert(0, G)
    return torch.tensor(returns)


def compute_advantages(returns, values, discount_factor=0.99):
    # Advantage A(s,a) = G_t - V(s)
    advantages = returns - values
    return advantages


def normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)