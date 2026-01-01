import pprint

import gymnasium as gym
import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env
import wandb

class CartPoleMLP(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CustomConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN for processing 96*96*3 RGB images
        # 96 -> 24 -> 12 -> 10 -> 8 -> 6 -> 4
        # channels: 3 -> 16 -> 32 -> 64 -> 128 -> 256
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)

    def forward(self, x):
        # x is expected to be (B, C, H, W) and normalized if handled outside,
        # but here we will handle normalization in the module to match the original logic
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        return x

class CustomCartPolePPORLModule(PPOTorchRLModule):

    def setup(self):
        # Feature extractor
        self.encoder = CartPoleMLP(obs_dim=4, hidden_dim=128)

        # Policy head (Discrete → logits)
        self.policy_head = nn.Linear(128, 2)

        # Value head
        self.vf_head = nn.Linear(128, 1)

    def get_initial_state(self):
        return {}

    def _get_features(self, batch):
        obs = batch[Columns.OBS].float()   # (B, 4)
        return self.encoder(obs)

    def _forward(self, batch, **kwargs):
        features = self._get_features(batch)

        logits = self.policy_head(features)
        values = self.vf_head(features).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }

    def compute_values(self, batch, **kwargs):
        features = self._get_features(batch)
        return self.vf_head(features).squeeze(-1)

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)



# --- Environment Setup ---

def make_car_racing_env(config):
    # No NormalizeImageWrapper because the model handles it
    env = gym.make("CartPole-v1")
    return env


register_env("CartPole-Custom", make_car_racing_env)

# --- Config and Training ---
wandb.init(
    entity="alienpenguin-inc",
    project="rl-training",
    name="ppo-baseline-CartPole-v1",
    config={
        "algorithm": "ppo-baseline",
        "environment": "CartPole-v1",
        "max_episodes": 1000,
        "gamma": 0.99,
        "kl_coeff": 0.2,
    }
)

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-Custom")
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=CustomCartPolePPORLModule,
        ),
    )
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size_per_learner=4000,
        minibatch_size=128,
        num_sgd_iter=10,
    )
    .resources(num_gpus=1)
)


algo = config.build()

print("Starting training with Custom CNN RLModule...")
num_iterations = 10000
num_saves = num_iterations / 10
for i in range(num_iterations):
    result = algo.train()
    episode_reward = result.get('env_runners').get('episode_return_mean')

    wandb.log({
        "reward_mean": episode_reward,
        "duration_sec_mean": result.get('env_runners').get('episode_duration_sec_mean'),
        "policy_loss": result.get('learners').get('default_policy').get('policy_loss'),
        "value_function_loss": result.get('learners').get('default_policy').get('vf_loss'),
    })

    print(f"Iteration {i + 1}: episode_reward_mean = {episode_reward}")

    if (i + 1) % num_saves == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved at: {checkpoint_dir}")

wandb.finish()