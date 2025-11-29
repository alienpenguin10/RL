import torch
import torch.nn as nn
import numpy as np

class BaseAgent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Detect device here to be used by all children
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.terminated = []
        self.truncated = []

    def store_transition(self, state, action, reward, log_prob, terminated, truncated, value=None):
        # Standardized signature
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.terminated.append(terminated) 
        self.truncated.append(truncated) 
        self.values.append(value)

    def preprocess_state(self, state):
        # Input: (96, 96, 3) -> Output:(1, 3, 96, 96) on DEVICE
        # Pytorch expects CHW format but state is HWC, so we need to permute the dimensions
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0)
        return state.to(self.device) # Crucial: Move to GPU

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