import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store transition. State should already be (4, 84, 84)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample batch and convert to tensors"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # States are already (4, 84, 84), no permute needed
        return (
            torch.FloatTensor(np.array(states)),  # (batch, 4, 84, 84)
            torch.FloatTensor(np.array(actions)),  # (batch, action_dim)
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),  # (batch, 1)
            torch.FloatTensor(np.array(next_states)),  # (batch, 4, 84, 84)
            torch.FloatTensor(np.array(dones)).unsqueeze(1)  # (batch, 1)
        )

    def __len__(self):
        return len(self.buffer)