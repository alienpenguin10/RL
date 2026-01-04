import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta


def layer_init(
    layer: nn.Linear | nn.Conv2d,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear | nn.Conv2d:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()

        dims = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            linear = layer_init(nn.Linear(dims[i], dims[i + 1]))
            layers.extend([linear, activation()])

        output_linear = layer_init(nn.Linear(dims[-1], output_dim))
        layers.append(output_linear)

        if output_activation:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        feature_dim: int = 512,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.conv_layers = nn.Sequential(
            layer_init(nn.Conv2d(self.obs_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *self.obs_shape)
            dummy = self.conv_layers(dummy)
            conv_out_size = dummy.numel()

        self.fc = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, feature_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        feature_dim: int = 512,
        hidden_dims: tuple[int, ...] = (256, 256),
        log_std_min: float = -20,
        log_std_max: float = 2,
    ) -> None:
        super().__init__()

        # Constants for log_std clamping
        # This is used to prevent the log_std from becoming too large or too small
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.cnn = ConvNet(obs_shape, feature_dim)
        self.mean_net = MultiLayerPerceptron(feature_dim, hidden_dims, action_dim)
        self.log_std_net = MultiLayerPerceptron(feature_dim, hidden_dims, action_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = self.cnn(x)
        mean = self.mean_net(output)
        log_std = torch.clamp(
            self.log_std_net(output), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(
        self, state: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        means, log_stds = self.forward(state)

        if deterministic:
            return torch.tanh(means), None, None

        stds = torch.exp(log_stds)
        normal = torch.distributions.Normal(means, stds)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)

        return action, log_prob, entropy

    @torch.no_grad()
    def act(self, state: Tensor, deterministic: bool = False) -> np.ndarray:
        action, _, _ = self.sample(state, deterministic)
        return action.cpu().numpy()

    def get_log_prob(self, states, actions):
        # means:(batch_size, 3) = [[steering_mean, gas_mean, brake_mean]], log_stds:(batch_size, 3) = [[steering_log_std, gas_log_std, brake_log_std]]
        means, log_stds = self.forward(states)

        # stds:(batch_size, 3) = [[steering_std, gas_std, brake_std]]
        stds = torch.exp(log_stds)

        # dist: Gaussian distribution with mean and std
        dist = torch.distributions.Normal(means, stds)

        # log_probs:(batch_size,) = [total_log_prob] = [log_prob_steering + log_prob_gas + log_prob_brake] = [log(Normal(0.2, 0.223).pdf(steering_raw)) + log(Normal(0.7, 0.135).pdf(gas_raw)) + log(Normal(0.1, 0.050).pdf(brake_raw))]
        log_probs = dist.log_prob(actions).sum(dim=1)

        return log_probs


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        feature_dim: int = 1024,
        hidden_dims: tuple[int, ...] = (512, 64),
    ) -> None:
        super().__init__()
        self.convnet = ConvNet(obs_shape, feature_dim)
        self.net = MultiLayerPerceptron(feature_dim, hidden_dims, 1)

    def forward(self, x):
        x = self.convnet(x)
        return self.net(x).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        feature_dim: int = 512,
        hidden_dims: tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        self.cnn = ConvNet(obs_shape, feature_dim)
        self.net = MultiLayerPerceptron(feature_dim + action_dim, hidden_dims, 1)

    def forward(self, state, action):
        features = self.cnn(state)
        return self.net(torch.cat([features, action], dim=-1))


class ActorCriticThreeOutput(nn.Module):
    def __init__(
        self,
        obs_shape=(4, 96, 96),
        action_dim=3,
        feature_dim=512,
        hidden_dims=[256, 256],
    ):
        super().__init__()
        self.state_dims = obs_shape
        self.action_dim = action_dim

        self.cnn = ConvNet(obs_shape, feature_dim)
        # actor outputs alpha and beta parameters for Beta distribution for each action dimension
        self.actor = layer_init(nn.Linear(feature_dim, action_dim * 2), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "Input must be a tensor"
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # To make sure state has a batch dimension

        x = self.cnn(x)

        # Get Alpha and Beta parameters
        # We use Softplus + 1.0 to ensure alpha, beta >= 1.0.
        # This constrains the Beta distribution to be unimodal (bell-shaped),
        # preventing the "U-shaped" bimodality that destabilizes training.
        policy_output = self.actor(x)
        alpha_beta = F.softplus(policy_output) + 1.0
        alpha, beta = torch.chunk(alpha_beta, 2, dim=-1)

        value = self.critic(x)

        # return steer_alpha, steer_beta, speed_alpha, speed_beta, value
        return alpha, beta, value

    def get_action(self, state):
        # Takes a single state -> samples a new action from policy dist
        alpha, beta, value = self.forward(state)
        dist = torch.distributions.Beta(alpha, beta)
        sample = dist.sample()

        # Affine Transformation
        # Map raw sample to environment bounds
        # Steering (idx 0): -> [-1, 1] via y = x*2-1
        # Gas (idx 1): -> [0, 1] via y = x
        # Brake (idx 2): -> [0, 1] via y = x
        action = (
            torch.stack(
                [
                    sample[:, 0] * 2 - 1,  # Steering
                    sample[:, 1],  # Gas
                    sample[:, 2],  # Brake
                ],
                dim=1,
            )
            .cpu()
            .numpy()
            .flatten()
        )

        # Log Prob Correction
        # When transforming a variable, we must correct the density.
        # For steering y = 2x - 1, dy/dx = 2.
        # log_prob(y) = log_prob(x) - log(|dy/dx|) = log_prob(x) - log(2)
        # This correction applies only to the steering dimension.
        log_prob_per_dim = dist.log_prob(sample)
        log_prob_per_dim[:, 0] -= torch.log(torch.tensor(2.0))
        log_prob = log_prob_per_dim.sum(dim=1).cpu().numpy().flatten()

        return action, log_prob, value

    def get_value(self, state):
        # Takes a single state -> returns value estimate
        _, _, value = self.forward(state)
        return value

    def evaluate(self, states, actions):
        # takes in batch of states and actions -> doesn't sample evaluates the log prob of specific action under the current policy
        # also returns entropy regularization term
        alpha, beta, value = self.forward(states)
        dist = torch.distributions.Beta(alpha, beta)

        # Inverse Transformation
        # The 'actions' passed here are from the reply buffer (Env space)
        # We must map them back to evaluate them under the Beta distribution
        # Inverse steering: [-1, 1] -> [0, 1]:  y = x*2-1 -> x = (y + 1) / 2
        # Inverse gas: [0, 1] -> [0, 1]: y = x
        # Inverse brake: [0, 1] -> [0, 1]: y = x
        sample_actions = torch.stack(
            [
                (actions[:, 0] + 1) / 2,  # Steering
                actions[:, 1],  # Gas
                actions[:, 2],  # Brake
            ],
            dim=1,
        )
        # Numerical stability: clamp to avoid exact 0 or 1 which can cause inf log_prob
        sample_actions = torch.clamp(sample_actions, 1e-6, 1.0 - 1e-6)
        log_prob_per_dim = dist.log_prob(sample_actions)
        # Apply log prob correction for scaling the steering dimension
        log_prob_per_dim[:, 0] -= torch.log(torch.tensor(2.0))
        log_prob = log_prob_per_dim.sum(dim=1)

        entropy = dist.entropy().sum(dim=1)

        return log_prob, value, entropy


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int] = (4, 96, 96),
        action_dim: int = 2,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        self.cnn = ConvNet(obs_shape, feature_dim)

        # actor outputs alpha and beta parameters for Beta distribution for each action dimension
        self.actor = layer_init(nn.Linear(feature_dim, action_dim * 2), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Make sure state has a batch dimension
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)

        obs = self.cnn(obs)

        # Get Alpha and Beta parameters
        # We use Softplus + 1.0 to ensure alpha, beta >= 1.0.
        # This constrains the Beta distribution to be unimodal (bell-shaped),
        # preventing the "U-shaped" bimodality that destabilizes training.
        policy_output = self.actor(obs)
        alpha_beta = F.softplus(policy_output) + 1.0
        alpha, beta = torch.chunk(alpha_beta, 2, dim=-1)

        value = self.critic(obs)
        return alpha, beta, value

    def _get_distribution(self, obs: Tensor) -> tuple[Beta, Tensor]:
        alpha, beta, value = self.forward(obs)
        return Beta(alpha, beta), value

    def _transform_action(self, sample: Tensor) -> np.ndarray:
        if self.action_dim == 2:
            action = (sample.cpu().numpy().flatten() * 2) - 1  # Scale to [-1, 1]
        elif self.action_dim == 3:
            action = (
                torch.stack(
                    [
                        sample[:, 0] * 2 - 1,  # Steering
                        sample[:, 1],  # Gas
                        sample[:, 2],  # Brake
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
                .flatten()
            )
        else:
            raise ValueError("Unsupported action dimension")
        return action

    def _inverse_transform_action(self, action: Tensor) -> Tensor:
        if self.action_dim == 2:
            # Inverse Transformation
            # The 'actions' passed here are from the reply buffer (Env space)
            # Rescale actions from [-1, 1] to [0, 1] for Beta distribution
            # y = x*2-1 -> x = (y + 1) / 2
            sample_action = torch.stack(
                [(action[:, 0] + 1) / 2, (action[:, 1] + 1) / 2], dim=1
            )
        elif self.action_dim == 3:
            sample_action = torch.stack(
                [
                    (action[:, 0] + 1) / 2,  # Steering
                    action[:, 1],  # Gas
                    action[:, 2],  # Brake
                ],
                dim=1,
            )
        else:
            raise ValueError("Unsupported action dimension")

        sample_action = torch.clamp(sample_action, 1e-6, 1.0 - 1e-6)
        return sample_action

    def _calculate_log_prob(self, dist: Beta, sample: Tensor) -> Tensor:
        if self.action_dim == 2:
            # Log Prob Correction
            # When transforming a variable, we must correct the density.
            # y = 2x - 1, dy/dx = 2.
            # log_prob(y) = log_prob(x) - log(|dy/dx|) = log_prob(x) - log(2)
            log_prob_per_dim = dist.log_prob(sample)
            log_prob_per_dim -= torch.log(torch.tensor(2.0))
            log_prob = log_prob_per_dim.sum(1)
        elif self.action_dim == 3:
            log_prob_per_dim = dist.log_prob(sample)
            # Apply log prob correction for scaling the steering dimension
            log_prob_per_dim[:, 0] -= torch.log(torch.tensor(2.0))
            log_prob = log_prob_per_dim.sum(dim=1)
        else:
            raise ValueError("Unsupported action dimension")
        return log_prob

    def get_action(self, obs: Tensor) -> tuple[np.ndarray, np.ndarray, Tensor]:
        # Takes a single state -> samples a new action from policy dist
        dist, value = self._get_distribution(obs)
        sample = dist.sample()

        action = self._transform_action(sample)
        log_prob = self._calculate_log_prob(dist, sample)

        return action, log_prob.cpu().numpy().flatten(), value

    def get_value(self, state: Tensor) -> Tensor:
        # Takes a single state -> returns value estimate
        _, _, value = self.forward(state)
        return value

    def evaluate(
        self,
        obs: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # takes in batch of states and actions -> doesn't sample evaluates the log prob of specific action under the current policy
        # also returns entropy regularization term
        dist, value = self._get_distribution(obs)
        samples = self._inverse_transform_action(action)

        log_prob = self._calculate_log_prob(dist, samples)
        entropy = dist.entropy().sum(dim=1)

        return log_prob, value, entropy
