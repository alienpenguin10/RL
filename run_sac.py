import argparse
import os
import signal
import sys
import time

import gymnasium as gym
import numpy as np
import yaml
from dotenv import load_dotenv
from gymnasium.wrappers import RecordVideo

from agents.sac import SACAgent
from env.wrappers import (
    ActionMapWrapper,
    ActionRepeatWrapper,
    FrameStackWrapper,
    PreprocessWrapper,
    RewardShapingWrapper,
)

# Load environment variables from .env file
load_dotenv()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training will continue without logging")


def create_env(
    env_name,
    use_grayscale,
    use_frame_stack,
    use_skip_frame,
    n_stack,
    use_throttle_action=True,
    record_video=False,
    video_folder=None,
    video_interval=10,
):
    render_mode = "rgb_array" if record_video else None
    env = gym.make(env_name, continuous=True, render_mode=render_mode)

    if use_throttle_action:
        env = ActionMapWrapper(env)

    env = RewardShapingWrapper(
        env,
        reward_scale=1,  # Scales 7.0 -> 0.35
        skip_zoom_steps=50,  # Skips the uncontrollable intro
        patience=100,  # Kills episode if stuck for 100 steps
    )

    if use_skip_frame:
        env = ActionRepeatWrapper(env, repeat=4)

    if record_video and video_folder:
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: episode_id % video_interval == 0,
            name_prefix="sac-eval",
            disable_logger=True,
        )

    if use_frame_stack or use_grayscale:
        env = PreprocessWrapper(env, resize=(84, 84), grayscale=use_grayscale)

    if use_frame_stack:
        env = FrameStackWrapper(env, n_stack, skip_frames=use_skip_frame and 4 or 0)

    return env


def train_sac_stepwise(config):
    # Extract config values
    env_name = config["env_id"]
    max_timesteps = config["max_timesteps"]
    start_steps = config["start_steps"]
    checkpoint_interval = config["checkpoint_interval"]
    use_grayscale = config["use_grayscale"]
    use_skip_frame = config.get("use_skip_frame", False)
    use_frame_stack = config["use_frame_stack"]
    use_throttle_action = config.get("use_throttle_action", False)
    n_stack = config.get("n_stack", 4)
    log_freq = config["log_freq"]
    max_ep_len = config["max_ep_len"]
    record_video = config.get("record_video", False)
    video_folder = config.get("video_folder", "./videos")
    video_interval = config.get("video_interval", 1)
    q_hidden_dims = config.get("q_hidden_dims", [256, 256])
    policy_hidden_dims = config.get("policy_hidden_dims", [256, 256])

    # Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project=config.get("project_name", "rl-training"),
            name=config.get("run_name", f"sac-{env_name}"),
            config=config,
        )

    env = create_env(
        env_name,
        use_grayscale,
        use_frame_stack,
        use_skip_frame,
        n_stack,
        use_throttle_action,
        record_video=False,
    )

    print("Environment Configuration:")
    print(f"  Use Grayscale: {use_grayscale}")
    print(f"  Use Frame Stack: {use_frame_stack} (n_stack={n_stack})")
    print(f"  Use Repeat Action: {use_skip_frame}")
    print(f"  Use Throttle Action: {use_throttle_action}")
    if use_throttle_action:
        print(f"    → Action space: [steering, throttle] (2D)")
    else:
        print(f"    → Action space: [steering, gas, brake] (3D)")
    print()

    # Create EVALUATION environment (with video recording)
    eval_env = create_env(
        env_name,
        use_grayscale,
        use_frame_stack,
        use_skip_frame,
        n_stack,
        use_throttle_action,
        record_video=record_video,
        video_folder=video_folder,
        video_interval=video_interval,
    )

    if record_video:
        print(
            f"Recording videos to: {video_folder} (every {video_interval} eval episodes)"
        )

    print(f"Environment observation shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")

    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        policy_lr=config["policy_learning_rate"],
        q_lr=config["q_learning_rate"],
        alpha_lr=config["alpha_learning_rate"],
        policy_weight_decay=config["policy_weight_decay"],
        q_weight_decay=config["q_weight_decay"],
        alpha_weight_decay=config["alpha_weight_decay"],
        q_hidden_dims=q_hidden_dims,
        policy_hidden_dims=policy_hidden_dims,
    )

    batch_size = agent.batch_size
    train_freq = config["train_freq"]
    gradient_steps = config["gradient_steps"]

    print(f"Training with batch size: {batch_size}")
    print(f"Device: {agent.device}")
    print(f"Train frequency: {train_freq}, Gradient steps: {gradient_steps}")

    current_step = [0]

    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model at step {current_step[0]}...")
        agent.save_model(f"./models/sac_step_{current_step[0]}_interrupted.pth")
        print("Model saved!")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, save_on_interrupt)

    # Episode tracking
    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards_history = []
    last_log_step = 0
    episode_steps = 0
    episode_num = 0
    cur_best_evaluation_score = -np.inf

    # Timing statistics
    start_time = time.time()

    print(f"\nStarting training for {max_timesteps:,} steps...")
    print(f"Warmup period: {start_steps:,} steps\n")

    # fill buffer
    if start_steps > 0:
        print(f"[BUFFER] Prefilling buffer ({start_steps} steps)...")
        buffer_rewards = []
        buffer_ep_reward = 0
        buffer_ep_count = 0
        report_buffer_progress_interval = max(start_steps // 10, 1000)

        for buffer_step in range(start_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.replay_buffer.push(
                obs, action, reward, next_obs, terminated or truncated
            )

            buffer_ep_reward += reward
            obs = next_obs

            if terminated or truncated:
                buffer_rewards.append(buffer_ep_reward)
                buffer_ep_count += 1
                obs, _ = env.reset()
                buffer_ep_reward = 0

            if (buffer_step + 1) % report_buffer_progress_interval == 0:
                progress = 100 * (buffer_step + 1) / start_steps
                buffer_size = len(agent.replay_buffer)
                avg_reward = np.mean(buffer_rewards) if buffer_rewards else 0
                print(
                    f"  [{progress:5.1f}%] Step {buffer_step + 1:,}/{start_steps:,} | "
                    f"Buffer: {buffer_size:,} | Episodes: {buffer_ep_count} | "
                    f"Avg Reward: {avg_reward:.2f}"
                )

        final_buffer_size = len(agent.replay_buffer)
        final_avg = np.mean(buffer_rewards) if buffer_rewards else 0
        print(
            f"[OK] Buffer prefilled: {final_buffer_size:,} transitions | "
            f"{buffer_ep_count} episodes | Avg reward: {final_avg:.2f}\n"
        )

    for total_steps in range(1, max_timesteps + 1):
        current_step[0] = total_steps

        if total_steps < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(obs, action, reward, next_obs, None, done)

        if WANDB_AVAILABLE and total_steps % 100 == 0:
            if use_throttle_action:
                wandb.log(
                    {
                        "action/steering": action[0],
                        "action/throttle": action[1],
                    },
                    step=total_steps,
                )
            else:
                wandb.log(
                    {
                        "action/steering": action[0],
                        "action/gas": action[1],
                        "action/brake": action[2],
                    },
                    step=total_steps,
                )

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        # Training update logic
        buffer_warmed_up = len(agent.replay_buffer) >= batch_size

        if (
            buffer_warmed_up
            and total_steps >= start_steps
            and total_steps % train_freq == 0
        ):
            metrics = None
            for _ in range(gradient_steps):
                metrics = agent.update()
            if metrics and WANDB_AVAILABLE:
                wandb.log(
                    {f"train/{k}": v for k, v in metrics.items()}, step=total_steps
                )

        # Handle episode end
        if done:
            episode_num += 1
            episode_rewards_history.append(episode_reward)

            if len(episode_rewards_history) > 100:
                episode_rewards_history.pop(0)

            success = episode_reward >= 900

            if WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/episode_reward": episode_reward,
                        "train/episode_length": episode_steps,
                        "train/episode_count": episode_num,
                        "train/success": int(success),
                        "train/rolling_mean_reward": np.mean(episode_rewards_history),
                        "train/rolling_std_reward": np.std(episode_rewards_history)
                        if len(episode_rewards_history) > 1
                        else 0,
                    },
                    step=total_steps,
                )

            if total_steps - last_log_step >= log_freq:
                progress = 100 * total_steps / max_timesteps
                recent_mean = (
                    np.mean(episode_rewards_history[-10:])
                    if episode_rewards_history
                    else 0
                )
                print(
                    f"[PROGRESS] Step {total_steps:,}/{max_timesteps:,} ({progress:.1f}%) | "
                    f"Episodes: {episode_num} | Recent Avg: {recent_mean:.1f}"
                )
                last_log_step = total_steps

            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0

        # Evaluation + Video Recording
        if total_steps % checkpoint_interval == 0 and total_steps > 0:
            print(f"\n[EVAL] Evaluating at step {total_steps:,}...")
            evaluation_rewards = []

            for evaluation_episode in range(5):
                evaluation_obs, _ = eval_env.reset()
                evaluation_done = False
                evaluation_ep_reward = 0
                evaluation_steps = 0

                while not evaluation_done and evaluation_steps < max_ep_len:
                    evaluation_action = agent.select_action(
                        evaluation_obs, deterministic=True
                    )
                    (
                        evaluation_obs,
                        step_reward,
                        evaluation_terminated,
                        evaluation_truncated,
                        _,
                    ) = eval_env.step(evaluation_action)
                    evaluation_done = evaluation_terminated or evaluation_truncated
                    evaluation_ep_reward += step_reward
                    evaluation_steps += 1

                evaluation_rewards.append(evaluation_ep_reward)
                print(
                    f"    Eval Episode {evaluation_episode + 1}: {evaluation_ep_reward:.2f}"
                )

            evaluation_mean = np.mean(evaluation_rewards)
            eval_std = np.std(evaluation_rewards)
            print(f"[EVAL] Mean reward: {evaluation_mean:.2f} ± {eval_std:.2f}")

            if WANDB_AVAILABLE:
                wandb.log(
                    {
                        "eval/mean_reward": evaluation_mean,
                        "eval/std_reward": eval_std,
                        "eval/min_reward": np.min(evaluation_rewards),
                        "eval/max_reward": np.max(evaluation_rewards),
                    },
                    step=total_steps,
                )

            if evaluation_mean > cur_best_evaluation_score:
                cur_best_evaluation_score = evaluation_mean
                agent.save_model(f"./models/sac_best.pth")
                print(
                    f"[BEST] New best model saved! Score: {cur_best_evaluation_score:.2f}"
                )

            agent.save_model(f"./models/sac_step_{total_steps}.pth")
            print(f"[CHECKPOINT] Saved checkpoint at step {total_steps:,}\n")

    # Save final model
    agent.save_model(f"./models/sac_final.pth")
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("Training complete! Final model saved.")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Total episodes: {episode_num}")
    print(f"Average time per 1000 steps: {total_time / (max_timesteps / 1000):.2f}s")
    print(f"{'=' * 60}")

    if WANDB_AVAILABLE:
        wandb.finish()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs("./models", exist_ok=True)

    print("\n--- Training SAC with step-based updates ---")
    print(f"Configuration: {args.config}")
    print(f"Environment: {config['env_id']}")
    print(f"Max timesteps: {config['max_timesteps']:,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Start steps: {config['start_steps']:,}")
    print(f"Use Throttle Action: {config.get('use_throttle_action', True)}")
    print(f"Hidden dims (Q-function): {config.get('q_hidden_dims', [256, 256])}")
    print(f"Hidden dims (Policy): {config.get('policy_hidden_dims', [256, 256])}")
    print(
        f"Learning Rates - Policy: {config['policy_learning_rate']}, Q-function: {config['q_learning_rate']}, Alpha: {config['alpha_learning_rate']}\n"
    )

    train_sac_stepwise(config)
