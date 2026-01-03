import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def clear(self):
        self.buffer.clear()

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float = None,
        next_state: np.ndarray = None,
        done: bool | None = None,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray = None,
        next_states: np.ndarray = None,
        dones: np.ndarray = None,
    ):
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "obs": torch.FloatTensor(np.array(states)).to(device),
            "actions": torch.FloatTensor(np.array(actions)).to(device),
            "rewards": torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            "next_obs": torch.FloatTensor(np.array(next_states)).to(device),
            "dones": torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        }

    def get_replay(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)
