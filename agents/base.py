import numpy as np
import torch
import torch.nn as nn
from typing_extensions import override


class BaseAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Detect device here to be used by all children
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.terminated = []
        self.truncated = []

    def store_transition(
        self, state, action, reward, log_prob, terminated, truncated, value=None
    ):
        # Standardized signature
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.values.append(value)

    def preprocess_state(self, state):
        if isinstance(state, np.ndarray):
            # Check if state is already in (C, H, W) format
            if state.shape[0] in [3, 4]:  # Channels-first (C, H, W)
                state = torch.from_numpy(state).float()
            elif state.shape[-1] in [3, 4]:  # Channels-last (H, W, C)
                state = np.transpose(state, (2, 0, 1))  # -> (C, H, W)
                state = torch.from_numpy(state).float()
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")

        if state.dim() == 3:
            state = state.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)

        return state.to(self.device)

    def save_model(self, filepath):
        """
        Saves the model to a file
        """
        raise NotImplementedError("save_model method not implemented in BaseAgent")

    def load_model(self, filepath):
        """
        Loads the model from a file
        """
        raise NotImplementedError("load_model method not implemented in BaseAgent")
