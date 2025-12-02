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


class CustomPPORLModule(PPOTorchRLModule):
    def setup(self):
        # Define the architecture
        self.convnet = CustomConvNet()

        # ConvNet output: 256 channels * 4 * 4 spatial = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)

        # Policy Heads
        # Steering: [-1, 1] (tanh)
        self.steering_mean = nn.Linear(64, 1)
        self.steering_log_std = nn.Linear(64, 1)

        # Gas: [0, 1] (sigmoid)
        self.gas_mean = nn.Linear(64, 1)
        self.gas_log_std = nn.Linear(64, 1)

        # Brake: [0, 1] (sigmoid)
        self.brake_mean = nn.Linear(64, 1)
        self.brake_log_std = nn.Linear(64, 1)

        # Value Head
        self.vf_head = nn.Linear(64, 1)

        # Log std bounds
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def get_initial_state(self):
        return {}

    def _get_features(self, batch):
        obs = batch[Columns.OBS]
        # Preprocess observation
        # Input obs is (B, H, W, C) uint8 [0, 255] usually
        # We need to convert to float, normalize, and permute to (B, C, H, W)
        x = obs.float() / 255.0
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # CNN
        x = self.convnet(x)
        x = x.reshape(x.size(0), -1)  # Flatten

        # FC Layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def _forward(self, batch, **kwargs):
        x = self._get_features(batch)

        # Value function
        vf_out = self.vf_head(x).squeeze(-1)

        # Policy Means
        steering_mean = torch.tanh(self.steering_mean(x))
        gas_mean = torch.sigmoid(self.gas_mean(x))
        brake_mean = torch.sigmoid(self.brake_mean(x))

        # Policy Log Stds
        steering_log_std = torch.clamp(self.steering_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        gas_log_std = torch.clamp(self.gas_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        brake_log_std = torch.clamp(self.brake_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)

        # Concatenate for action distribution
        means = torch.cat([steering_mean, gas_mean, brake_mean], dim=1)
        log_stds = torch.cat([steering_log_std, gas_log_std, brake_log_std], dim=1)

        action_dist_inputs = torch.cat([means, log_stds], dim=1)

        return {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.VF_PREDS: vf_out,
        }

    def compute_values(self, batch, **kwargs):
        x = self._get_features(batch)
        return self.vf_head(x).squeeze(-1)

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, **kwargs)


# --- Environment Setup ---

def make_car_racing_env(config):
    # No NormalizeImageWrapper because the model handles it
    env = gym.make("CarRacing-v3")
    return env


register_env("CarRacing-Custom", make_car_racing_env)

# --- Config and Training ---
wandb.init(
    entity="alienpenguin-inc",
    project="rl-training",
    name="ppo-baseline-CarRacing-v3",
    config={
        "algorithm": "ppo-baseline",
        "environment": "CarRacing-v3",
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
    .environment("CarRacing-Custom")
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=CustomPPORLModule,
        ),
    )
    .training(
        gamma=0.99,
        lr=0.0003,
        kl_coeff=0.2,
        train_batch_size_per_learner=4000,
        minibatch_size=128,
        num_sgd_iter=10,
    )
    .resources(num_gpus=1)  # Set to 1 if GPU is available
)

algo = config.build()

print("Starting training with Custom CNN RLModule...")
num_iterations = 1000000
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