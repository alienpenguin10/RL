import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device='cpu'):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            'obs': torch.FloatTensor(np.array(states)).to(device),
            'actions': torch.FloatTensor(np.array(actions)).to(device),
            'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            'next_obs': torch.FloatTensor(np.array(next_states)).to(device),
            'dones': torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        }

    def __len__(self):
        return len(self.buffer)