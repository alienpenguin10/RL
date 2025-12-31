import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override

class TorchBeta(TorchDistributionWrapper):
    """Beta distribution for continuous actions in [0, 1]."""
    @override(TorchDistributionWrapper)
    def __init__(self, inputs, model):
        # inputs shape: (batch_size, action_dim * 2)
        # Split into alpha and beta parameters
        alpha, beta = torch.chunk(inputs, 2, dim=-1)
        
        # Ensure alpha, beta > 1 for unimodal distribution
        self.alpha = F.softplus(alpha) + 1.0
        self.beta = F.softplus(beta) + 1.0
        
        # Create Beta distribution
        self.dist = torch.distributions.Beta(
            concentration1=self.alpha,
            concentration0=self.beta
        )
        super().__init__(inputs, model)
    
    @override(TorchDistributionWrapper)
    def deterministic_sample(self):
        # Return mean for deterministic action
        return self.dist.mean
    
    @override(TorchDistributionWrapper)
    def sample(self):
        # Sample from Beta distribution
        return self.dist.rsample()
    
    @override(TorchDistributionWrapper)
    def logp(self, actions):
        # Compute log probability
        return self.dist.log_prob(actions).sum(-1)
    
    @override(TorchDistributionWrapper)
    def entropy(self):
        return self.dist.entropy().sum(-1)
    
    @override(TorchDistributionWrapper)
    def kl(self, other):
        return torch.distributions.kl_divergence(
            self.dist, other.dist
        ).sum(-1)
    
    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        # Need 2 parameters (alpha, beta) per action dimension
        return action_space.shape[0] * 2
    
class TorchBetaTransformed(TorchBeta):
    """Beta distribution with affine transformations for different action ranges."""
    
    @override(TorchBeta)
    def sample(self):
        sample = super().sample()
        return self._transform(sample)
    
    @override(TorchBeta)
    def deterministic_sample(self):
        mean = super().deterministic_sample()
        return self._transform(mean)
    
    @override(TorchBeta)
    def logp(self, actions):
        # Inverse transform actions
        sample_actions = self._inverse_transform(actions)
        sample_actions = torch.clamp(sample_actions, 1e-6, 1.0 - 1e-6)
        
        # Get log prob from base distribution
        log_prob = self.dist.log_prob(sample_actions)
        
        # Apply Jacobian correction for transformations
        log_prob[:, 0] -= torch.log(torch.tensor(2.0))  # Steering correction
        
        return log_prob.sum(-1)
    
    def _transform(self, sample):
        # steering: [0,1] -> [-1,1], gas/brake: [0,1] -> [0,1]
        transformed = sample.clone()
        transformed[:, 0] = sample[:, 0] * 2 - 1  # Steering
        return transformed
    
    def _inverse_transform(self, actions):
        # steering: [-1,1] -> [0,1], gas/brake: [0,1] -> [0,1]
        sample = actions.clone()
        sample[:, 0] = (actions[:, 0] + 1) / 2  # Steering
        return sample