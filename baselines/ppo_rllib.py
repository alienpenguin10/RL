
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.networks import ConvNet
import ray
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticRLModule(DefaultPPOTorchRLModule):
    def setup(self):
        # You have access here to the following already set attributes:
        # self.observation_space
        # self.action_space
        # self.inference_only
        # self.model_config  # <- a dict with custom settings
        self.policy_output_dim = self.model_config.get("policy_output_dim", 2)
        self.feature_dim = self.model_config.get("feature_dim", 512)
        self.hidden_dim = self.model_config.get("hidden_dim", (256, 256))

        self.cnn = ConvNet(self.observation_space.shape, self.feature_dim)
        # actor outputs alpha and beta parameters for Beta distribution for each action dimension
        self.actor = layer_init(nn.Linear(self.feature_dim, self.policy_output_dim*2), std=0.01)
        self.critic = layer_init(nn.Linear(self.feature_dim, 1), std=1.0)
    
    def _forward(self, batch, **kwargs):
        x = self.cnn(batch[Columns.OBS])
        policy_output = self.actor(x)
        alpha_beta = F.softplus(policy_output) + 1.0
        value = self.critic(x).squeeze(-1)
        return {Columns.ACTION_DIST_INPUTS: alpha_beta, Columns.VF_PREDS: value}

    def compute_values(self, batch, **kwargs):
        x = self.cnn(batch[Columns.OBS])
        return self.critic(x).squeeze(-1)
    
    # def get_action(self, state):
    #     # Takes a single state -> samples a new action from policy dist
    #     alpha, beta, value = self.forward(state)
    #     dist = torch.distributions.Beta(alpha, beta)
    #     sample = dist.sample()

    #     action = (sample.cpu().numpy().flatten() * 2) - 1  # Scale to [-1, 1]

    #     # Log Prob Correction
    #     # When transforming a variable, we must correct the density.
    #     # y = 2x - 1, dy/dx = 2.
    #     # log_prob(y) = log_prob(x) - log(|dy/dx|) = log_prob(x) - log(2)
    #     log_prob_per_dim = dist.log_prob(sample)
    #     log_prob_per_dim -= torch.log(torch.tensor(2.0))
    #     log_prob = log_prob_per_dim.sum(1).cpu().numpy().flatten()

    #     return action, log_prob, value
    
    # def evaluate(self, states, actions):
    #     # takes in batch of states and actions -> doesn't sample evaluates the log prob of specific action under the current policy
    #     # also returns entropy regularization term
    #     alpha, beta, value = self.forward(states)
    #     dist = torch.distributions.Beta(alpha, beta)
        
    #     # Inverse Transformation
    #     # The 'actions' passed here are from the reply buffer (Env space)
    #     # Rescale actions from [-1, 1] to [0, 1] for Beta distribution
    #     # y = x*2-1 -> x = (y + 1) / 2
    #     sample_actions = torch.stack(
    #         [(actions[:, 0] + 1) / 2,
    #          (actions[:, 1] + 1) / 2],
    #         dim=1)
    #     # Numerical stability: clamp to avoid exact 0 or 1 which can cause inf log_prob
    #     sample_actions = torch.clamp(sample_actions, 1e-6, 1.0 - 1e-6)
    #     log_prob_per_dim = dist.log_prob(sample_actions)
    #     # Log Prob Correction for scaling
    #     log_prob_per_dim -= torch.log(torch.tensor(2.0)) 
    #     log_prob = log_prob_per_dim.sum(dim=1)
        
    #     entropy = dist.entropy().sum(dim=1)
        
    #     return log_prob, value, entropy
    
    def get_initial_state(self):
        return {}

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

