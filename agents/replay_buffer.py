import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1_000_000, obs_shape=None, action_dim=None):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self._obs_shape = obs_shape
        self._action_dim = action_dim
        self._initialized = False

        if obs_shape is not None and action_dim is not None:
            self._initialize_buffers(obs_shape, action_dim)

    def _initialize_buffers(self, obs_shape, action_dim):
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._initialized = True

    def push(self, obs, action, reward, next_obs, done):
        if not self._initialized:
            obs_shape = obs.shape
            action_dim = action.shape[0] if hasattr(action, 'shape') else len(action)
            self._initialize_buffers(obs_shape, action_dim)

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)

        use_pin_memory = device.type == 'cuda' if hasattr(device, 'type') else 'cuda' in str(device)

        def to_tensor(arr):
            tensor = torch.from_numpy(arr)
            if use_pin_memory:
                tensor = tensor.pin_memory()
            return tensor.to(device, non_blocking=use_pin_memory)

        return {
            'obs': to_tensor(self.obs[idxs]),
            'actions': to_tensor(self.actions[idxs]),
            'rewards': to_tensor(self.rewards[idxs]),
            'next_obs': to_tensor(self.next_obs[idxs]),
            'dones': to_tensor(self.dones[idxs]),
        }

    def __len__(self):
        return self.size
