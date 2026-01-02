import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device, gamma, lam):
        self.buffer_size = buffer_size
        self.device = device
        self.gamma = gamma
        self.lam = lam

        # Pre-allocate memory for the buffer
        self.states = torch.zeros((buffer_size, *state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.log_probs = torch.zeros((buffer_size, 1), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.dones = torch.zeros((buffer_size, 1), device=device)
        self.values = torch.zeros((buffer_size, 1), device=device)
        
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, state, action, log_prob, reward, done, value):
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full. Call finish_path and get() before storing more.")
        self.states[self.ptr] = state
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value

        self.ptr += 1

    def finish_path(self, last_value=0):
        # called at the end of a trajectory, or when the buffer is full
        # last_value: value of the next state (if not done), or 0 (if done)
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # Squeeze 2D tensors to 1D for calculations
        rewards_1d = self.rewards[path_slice].squeeze(-1)
        values_1d = self.values[path_slice].squeeze(-1)
        
        # Handle last_value - ensure it's a scalar tensor
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.squeeze().item() if last_value.numel() == 1 else last_value.item()
        
        rewards = torch.cat((rewards_1d, torch.tensor([last_value], device=self.device)))
        values = torch.cat((values_1d, torch.tensor([last_value], device=self.device)))

        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        # Efficient vector calculation of GAE
        # We need to reverse, compute cumsum, then reverse back
        # GAE[t] = delta[t] + (gamma * lambda) * GAE[t+1]
        # Ideally we compute this iteratively backwards to match standard GAE exactly
        path_len = self.ptr - self.path_start_idx
        advs = torch.zeros(path_len, device=self.device)
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            last_gae_lam = deltas[t] + self.gamma * self.lam * last_gae_lam
            advs[t] = last_gae_lam

        # Determine the target values (returns) for the value network
        # Returns = Advantages + Values
        returns = advs + values_1d

        # We store these temporarily to be retrieved by get()
        # Note: We don't overwrite self.rewards/values here to keep the raw data intact if needed, 
        # but typically we just need advantages and returns.
        if not hasattr(self, 'advantages'):
            self.advantages = torch.zeros(self.buffer_size, device=self.device)
            self.returns = torch.zeros(self.buffer_size, device=self.device)
            
        self.advantages[path_slice] = advs
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr

    def get(self):
        # reset buffer and reutnr data for training
        assert self.ptr == self.buffer_size, "Buffer not full yet!"
        self.ptr = 0
        self.path_start_idx = 0

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            log_probs=self.log_probs.squeeze(-1),  # Squeeze to 1D for consistency
            values=self.values.squeeze(-1)  # Squeeze to 1D for consistency
        )
        return data