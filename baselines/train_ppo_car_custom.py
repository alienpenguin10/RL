from env.wrappers import ActionMapWrapper, FrameStackWrapper, PreprocessWrapper

print("Script starting...")
import os
import sys

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
from dotenv import load_dotenv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

import wandb

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to allow importing from agents and env_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.networks import ConvNet_StackedFrames


class CustomPPORLModule(PPOTorchRLModule):
    def setup(self):
        # Define the architecture
        self.convnet = ConvNet_StackedFrames(num_frames=4)

        # ConvNet output: 256 channels * 4 * 4 spatial = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)

        # Policy Heads: ALL use Tanh now
        # Steering
        self.steering_mean = nn.Linear(64, 1)  # Outputting [-1, 1]
        self.steering_log_std = nn.Linear(64, 1)

        # Gas
        self.gas_mean = nn.Linear(64, 1)  # Outputting [-1, 1]
        self.gas_log_std = nn.Linear(64, 1)

        # Brake
        self.brake_mean = nn.Linear(64, 1)  # Outputting [-1, 1]
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
        # Input obs is (B, num_frames, H, W) from FrameStack
        # ConvNet_StackedFrames handles normalization internally

        # CNN
        x = self.convnet(obs)
        x = x.reshape(x.size(0), -1)  # Flatten

        # FC Layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def _forward(self, batch, **kwargs):
        x = self._get_features(batch)

        # Value function
        vf_out = self.vf_head(x).squeeze(-1)

        # ALL heads use Tanh
        steering_mean = torch.tanh(self.steering_mean(x))
        gas_mean = torch.tanh(self.gas_mean(x))
        brake_mean = torch.tanh(self.brake_mean(x))

        # Policy Log Stds
        steering_log_std = torch.clamp(
            self.steering_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        gas_log_std = torch.clamp(
            self.gas_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        brake_log_std = torch.clamp(
            self.brake_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX
        )

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
    env = gym.make("CarRacing-v3", continuous=True)
    env = PreprocessWrapper(env)
    env = ActionMapWrapper(env)
    env = FrameStackWrapper(
        env, num_frames=4, skip_frames=2
    )  # 2 frames stacked, skip 2
    return env


register_env("CarRacing-Custom", make_car_racing_env)

# --- Config and Training ---

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CarRacing-Custom")
    .env_runners(
        num_env_runners=8,  # n_envs: 8
        num_envs_per_env_runner=1,
        rollout_fragment_length=512,  # n_steps: 512
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=CustomPPORLModule,
        ),
    )
    .training(
        gamma=0.99,  # gamma
        lr=3e-4,  # learning_rate
        kl_coeff=0.2,
        clip_param=0.2,  # clip_range
        train_batch_size_per_learner=4096,  # batch_size (total) = 8 * 512
        minibatch_size=128,  # batch_size (sgd)
        num_sgd_iter=10,  # n_epochs
        grad_clip=0.5,  # max_grad_norm
        vf_loss_coeff=0.5,  # vf_coef
        entropy_coeff=0.01,  # ent_coef
        lambda_=0.95,  # gae_lambda
    )
    .resources(num_gpus=1)
)

if __name__ == "__main__":
    # Check for WandB API key
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError(
            "WANDB_API_KEY is not set in the environment variables. Please create a .env file with your WandB API key."
        )

    wandb.login(key=wandb_api_key)
    wandb.init(
        project="rl-training", name="ppo-car-custom-stacked", config=config.to_dict()
    )

    # Initialize Ray with runtime_env to ensure workers can find modules in parent directory
    # regardless of where the script is run from.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # We append to existing PYTHONPATH if it exists, or set it.
    current_pythonpath = os.getenv("PYTHONPATH", "")
    new_pythonpath = (
        f"{project_root}:{current_pythonpath}" if current_pythonpath else project_root
    )

    ray.init(runtime_env={"env_vars": {"PYTHONPATH": new_pythonpath}})

    algo = config.build()

    # Create checkpoint directory
    checkpoint_dir_base = os.path.join(os.getcwd(), "models", "train_ppo_car_custom")
    os.makedirs(checkpoint_dir_base, exist_ok=True)

    print("Starting training with Custom CNN RLModule...")
    for i in range(100):
        result = algo.train()

        # --- 1. Extract Environment Metrics ---
        env_runners_results = result.get("env_runners", {})
        episode_reward = env_runners_results.get("episode_return_mean", "N/A")
        episode_len = env_runners_results.get("episode_len_mean", "N/A")

        print(f"Iteration {i + 1}: episode_reward_mean = {episode_reward}")

        # --- 2. Extract Learner/Loss Metrics ---
        # Located in result['learners']['default_policy']
        learners_results = result.get("learners", {})
        default_policy_results = learners_results.get("default_policy", {})

        log_dict = {
            "iteration": i + 1,
            # Env Metrics
            "episode_reward_mean": episode_reward,
            "episode_len_mean": episode_len,
            # Time Metrics (Located at root)
            "training_iteration_time_ms": result.get("time_this_iter_s", 0) * 1000,
            # Learner Metrics (Losses, Entropy, etc.)
            "entropy": default_policy_results.get("entropy"),
            "total_loss": default_policy_results.get("total_loss"),
            "policy_loss": default_policy_results.get("policy_loss"),
            "vf_loss": default_policy_results.get("vf_loss"),
            "mean_kl_loss": default_policy_results.get("mean_kl_loss"),
        }

        # Filter out None values in case metrics aren't ready yet
        log_dict = {k: v for k, v in log_dict.items() if v is not None}

        wandb.log(log_dict)

        if (i + 1) % 10 == 0:
            checkpoint_path = algo.save(checkpoint_dir_base)
            print(f"Checkpoint saved at: {checkpoint_path}")

    wandb.finish()
