import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100_000, obs_shape=(4, 84, 84), action_dim=3, is_image=True):
        self.capacity = capacity
        self.is_image = is_image
        self.size = 0
        self.ptr = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8 if is_image else np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8 if is_image else np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        if self.is_image:
            self.obs[self.ptr] = (state * 255).astype(np.uint8)
            self.next_obs[self.ptr] = (next_state * 255).astype(np.uint8)
        else:
            self.obs[self.ptr] = state
            self.next_obs[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, batch_size)
        obs_batch = self.obs[idxs]
        next_obs_batch = self.next_obs[idxs]

        if self.is_image:
            obs_batch = obs_batch.astype(np.float32) / 255.0
            next_obs_batch = next_obs_batch.astype(np.float32) / 255.0

        batch = {
            'obs': torch.as_tensor(obs_batch, dtype=torch.float32, device=device),
            'actions': torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=device),
            'rewards': torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device),
            'next_obs': torch.as_tensor(next_obs_batch, dtype=torch.float32, device=device),
            'dones': torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device)
        }
        return batch

    def __len__(self):
        return self.size