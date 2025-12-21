print("Script starting...")
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env

import sys
import os
from dotenv import load_dotenv
import ray

import wandb

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to allow importing from agents and env_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.networks import ConvNet_StackedFrames
from env_wrapper import ProcessedFrame, FrameStack

import gymnasium as gym
import numpy as np


class ActionRemapWrapper(gym.ActionWrapper):
    """
    Agent acts in [-1, 1]^2:
      [steer, pedal]
    We remap to CarRacing continuous action space:
      steer: [-1, 1]
      gas:   [0, 1]
      brake: [0, 1]

    pedal > 0 -> gas = pedal, brake = 0
    pedal < 0 -> gas = 0, brake = -pedal
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def action(self, a):
        a = np.asarray(a, dtype=np.float32)
        steer = np.clip(a[0], -1.0, 1.0)

        pedal = np.clip(a[1], -1.0, 1.0)

        gas = np.clip(pedal, 0.0, 1.0)
        brake = np.clip(-pedal, 0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class CustomPPORLModule(PPOTorchRLModule):
    def setup(self):
        self.convnet = ConvNet_StackedFrames(num_frames=4)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)

        # Two action dimensions: steer + pedal
        self.steer_mean = nn.Linear(64, 1)
        self.steer_log_std = nn.Linear(64, 1)

        self.pedal_mean = nn.Linear(64, 1)
        self.pedal_log_std = nn.Linear(64, 1)

        self.vf_head = nn.Linear(64, 1)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def get_initial_state(self):
        return {}

    def _get_features(self, batch):
        obs = batch[Columns.OBS]
        x = self.convnet(obs)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def _forward(self, batch, **kwargs):
        x = self._get_features(batch)

        vf_out = self.vf_head(x).squeeze(-1)

        steer_mean = self.steer_mean(x)
        pedal_mean = self.pedal_mean(x)

        steer_log_std = torch.clamp(self.steer_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        pedal_log_std = torch.clamp(self.pedal_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)

        means = torch.cat([steer_mean, pedal_mean], dim=1)  # (B, 2)
        log_stds = torch.cat([steer_log_std, pedal_log_std], dim=1)  # (B, 2)

        action_dist_inputs = torch.cat([means, log_stds], dim=1)  # (B, 4)

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
    env = ProcessedFrame(env)
    env = ActionRemapWrapper(env)
    env = ActionRepeatWrapper(env, 4)  # execute same action for 4 steps
    env = FrameStack(env, num_frames=4, skip_frames=2)  # 2 frames stacked, skip 2
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
            "WANDB_API_KEY is not set in the environment variables. Please create a .env file with your WandB API key.")

    wandb.login(key=wandb_api_key)
    wandb.init(
        entity="alienpenguin-inc",
        project="rl-training",
        name="ppo-car-custom-stacked",
        config=config.to_dict()
    )

    # Initialize Ray with runtime_env to ensure workers can find modules in parent directory
    # regardless of where the script is run from.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # We append to existing PYTHONPATH if it exists, or set it.
    current_pythonpath = os.getenv("PYTHONPATH", "")
    new_pythonpath = f"{project_root}:{current_pythonpath}" if current_pythonpath else project_root

    ray.init(runtime_env={"env_vars": {"PYTHONPATH": new_pythonpath}})

    algo = config.build()

    # Create checkpoint directory
    checkpoint_dir_base = os.path.join(os.getcwd(), "models", "train_ppo_car_custom")
    os.makedirs(checkpoint_dir_base, exist_ok=True)

    print("Starting training with Custom CNN RLModule...")
    for i in range(100):
        result = algo.train()

        # --- 1. Extract Environment Metrics ---
        env_runners_results = result.get('env_runners', {})
        episode_reward = env_runners_results.get('episode_return_mean', 'N/A')
        episode_len = env_runners_results.get('episode_len_mean', 'N/A')

        print(f"Iteration {i + 1}: episode_reward_mean = {episode_reward}")

        # --- 2. Extract Learner/Loss Metrics ---
        # Located in result['learners']['default_policy']
        learners_results = result.get('learners', {})
        default_policy_results = learners_results.get('default_policy', {})

        log_dict = {
            "iteration": i + 1,
            # Env Metrics
            "episode_reward_mean": episode_reward,
            "episode_len_mean": episode_len,

            # Time Metrics (Located at root)
            "training_iteration_time_ms": result.get('time_this_iter_s', 0) * 1000,

            # Learner Metrics (Losses, Entropy, etc.)
            "entropy": default_policy_results.get('entropy'),
            "total_loss": default_policy_results.get('total_loss'),
            "policy_loss": default_policy_results.get('policy_loss'),
            "vf_loss": default_policy_results.get('vf_loss'),
            "mean_kl_loss": default_policy_results.get('mean_kl_loss'),
        }

        # Filter out None values in case metrics aren't ready yet
        log_dict = {k: v for k, v in log_dict.items() if v is not None}

        wandb.log(log_dict)

        if (i + 1) % 10 == 0:
            checkpoint_path = algo.save(checkpoint_dir_base)
            print(f"Checkpoint saved at: {checkpoint_path}")

    wandb.finish()

