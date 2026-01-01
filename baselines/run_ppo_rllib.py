import signal
import sys
import os
# Add parent directory to path to allow importing from agents and env_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import gymnasium as gym
from agents.ppo import PPOAgent
from agents.recording_buffer import RecordingBuffer
from CarRacingEnv.env_wrapper import ProcessedFrame, FrameStack, SpeedInfoWrapper, ActionRepeatWrapper, PolicyActionMapWrapper
from baselines.ppo_rllib import ActorCriticRLModule
from baselines.rllib_beta_dist import TorchBetaTransformed
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training will continue without logging")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# --- Environment Setup ---
def make_car_racing_env(config):
    env = gym.make("CarRacing-v3", continuous=True)
    env = ProcessedFrame(env)
    env = FrameStack(env, num_frames=4, skip_frames=2)  # 2 frames stacked, skip 2
    env = ActionRepeatWrapper(env, 4)  # execute same action for 4 steps
    env = PolicyActionMapWrapper(env)
    return env

register_env("CarRacing-Custom", make_car_racing_env)

# --- Register Custom Beta Distribution ---
ModelCatalog.register_custom_action_dist("beta", TorchBetaTransformed)

# --- Config and Training ---
config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CarRacing-Custom")
    .env_runners(
        num_env_runners=1,  # n_envs: 8
        num_envs_per_env_runner=1,
        rollout_fragment_length=512,  # n_steps: 512
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=ActorCriticRLModule,
        ),
        model_config_dict={
            "custom_action_dist": "beta",
            "policy_output_dim": 3,  # steering, gas, brake
        },
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

def train(log_wandb=False):
    print ("Starting training...")
    # Initialize WandB if available
    if WANDB_AVAILABLE and log_wandb:
        wandb.init(
            project="rl-training",
            name=f"ppo-rllib-clean",
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

        if WANDB_AVAILABLE and log_wandb:
            wandb.log(log_dict)

    if WANDB_AVAILABLE and log_wandb:
        wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    train(log_wandb=False)