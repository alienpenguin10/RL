import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = int(capacity)