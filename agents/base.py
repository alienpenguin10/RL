import torch
import torch.nn as nn
import numpy as np

class BaseAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Detect device here to be used by all children
<<<<<<< Updated upstream
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
=======

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("mps")
>>>>>>> Stashed changes
        
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []  # Added 'dones' which is needed for VPG/PPO

    def store_transition(self, state, action, reward, next_state, log_prob, done, value=None):
        # Standardized signature
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done) # Important for handling episode ends
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
        save_dict = {}
        if hasattr(self, 'policy_network'):
            save_dict['policy_network'] = self.policy_network.state_dict()
        if hasattr(self, 'value_network'):
            save_dict['value_network'] = self.value_network.state_dict()
        torch.save(save_dict, filepath)

    def load_model(self, filepath):
        """
        Loads the model from a file
        """
        checkpoint = torch.load(filepath)
        if hasattr(self, 'policy_network') and 'policy_network' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['policy_network'])
        if hasattr(self, 'value_network') and 'value_network' in checkpoint:
            self.value_network.load_state_dict(checkpoint['value_network'])