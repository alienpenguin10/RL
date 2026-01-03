import argparse
import os
import signal
import sys

import gymnasium as gym
import numpy as np
import torch
import yaml

from agents.ppo import PPOAgent
from agents.replay_buffer import ReplayBuffer
from env.wrappers import (
    ActionMapWrapper,
    ActionRepeatWrapper,
    FrameStackWrapper,
    PreprocessWrapper,
    SpeedInfoWrapper,
)

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


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_env(
    env_name="CarRacing-v3",
    render_env=False,
    use_grayscale=True,
    use_repeat_action=False,
    use_policy_action_map=True,
    policy_outputs=3,
    frame_stack=8,
    frame_skip=2,
    use_promote_speed=False,
):
    render_mode = "human" if render_env else None
    env = gym.make(env_name, continuous=True, render_mode=render_mode)
    if use_grayscale:
        env = PreprocessWrapper(env)
    env = (
        FrameStackWrapper(env, num_frames=frame_stack, skip_frames=frame_skip)
        if frame_stack > 1
        else env
    )
    if use_repeat_action:
        env = ActionRepeatWrapper(env, skip_frames=frame_skip)
    if use_policy_action_map:
        env = ActionMapWrapper(env, use_throttle=policy_outputs == 2)
    if use_promote_speed:
        env = SpeedInfoWrapper(env)

    return env


def train(device, config):
    env_name = config["env_id"]
    use_repeat_action = config.get("use_repeat_action", False)
    total_timesteps = config["max_timesteps"]
    horizon = config["buffer_size"]
    frame_skip = config["frame_skip"]
    policy_outputs = config["policy_outputs"]
    use_episode_cutoff = config["use_episode_cutoff"]
    episode_cutoff = config["episode_cutoff_steps"]
    cutoff_penalty = config["episode_cutoff_penalty"]
    use_truncated_penalty = config["use_truncated_penalty"]
    truncated_penalty = config["truncated_penalty"]
    save_recordings = config["save_recordings"]
    recording_threshold = config["recording_threshold"]
    save_checkpoints = config["save_checkpoints"]
    checkpoint_interval = config["checkpoint_interval"]
    log_wandb = config["log_wandb"]

    if WANDB_AVAILABLE and log_wandb:
        wandb.init(
            project=config.get("project_name", "rl-training"),
            name=config.get("run_name", f"ppo_{env_name}"),
            config=config,
        )

    env = create_env(
        env_name=env_name,
        render_env=config.get("render_environment", False),
        use_grayscale=config.get("use_grayscale", True),
        use_repeat_action=use_repeat_action,
        use_policy_action_map=config.get("use_policy_action_map", True),
        policy_outputs=config.get("policy_outputs", 3),
        frame_stack=config.get("frame_stack", 8),
        frame_skip=config.get("frame_skip", 2),
        use_promote_speed=config.get("use_promote_speed", False),
    )

    agent = PPOAgent(
        device,
        env,
        policy_outputs=policy_outputs,
        num_epochs=config.get("num_epochs", 4),
        batch_size=config.get("batch_size", 512),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        value_coef=config.get("value_coef", 0.01),
        epsilons=config.get("epsilons", 0.2),
        learning_rate=config.get("learning_rate", 3e-4),
        use_lr_scheduler=config.get("use_lr_scheduler", False),
        lr_updates=int(total_timesteps / horizon),
        entropy_coef=config.get("entropy_coef", 0.02),
        entropy_decay=config.get("entropy_decay", 1.0),
        l2_reg=config.get("weight_decay", 1e-2),
        max_grad_norm=config.get("max_grad_norm", 0.0),
        process_action_decay=config.get("process_action_decay", 0.99),
    )

    state, info = env.reset()
    state = torch.Tensor(state).to(device)
    total_steps = 0
    update_count = 0
    episode = 0  # Track current episode for saving on interrupt
    checkpoint = 0
    episode_steps = 0
    episode_lengths = []
    episode_reward = 0
    episode_rewards = []
    carryover_reward = 0
    episode_recording = ReplayBuffer(capacity=1000)  # episode is max 1000 steps

    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model from episode {episode}...")
        agent.save_model(f"./models/ppo_{episode}_interrupted.pth")
        print(f"Model saved to ./models/ppo_{episode}_interrupted.pth")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)

    def save_episode_recording(episode_num, episode_reward, recording_buffer):
        """Save episode recording to NPZ file"""
        replay_path = f"./models/replay/ppo_replay_ep{episode_num}_reward{int(episode_reward)}.npz"
        states, actions, _, _, _ = recording_buffer.get_replay()
        np.savez_compressed(replay_path, states=states, actions=actions)
        print(f"Replay saved to {replay_path}")

    while total_steps < total_timesteps:
        states = torch.zeros((horizon, *env.observation_space.shape)).to(device)
        actions = torch.zeros((horizon, policy_outputs)).to(device)
        rewards = torch.zeros((horizon)).to(device)
        terminateds = torch.zeros((horizon)).to(device)
        truncateds = torch.zeros((horizon)).to(device)
        values = torch.zeros((horizon)).to(device)
        log_probs = torch.zeros((horizon)).to(device)

        # Initialize episode reward with carryover from previous rollout
        episode_reward += carryover_reward

        # Rollout
        for step in range(horizon):
            states[step] = state
            with torch.no_grad():
                policy_action, log_prob, value = agent.policy.get_action(state)
                values[step] = torch.tensor(value).to(device)
            actions[step] = torch.tensor(policy_action).to(device)
            log_probs[step] = torch.tensor(log_prob).to(device)
            processed_action = agent.process_action(policy_action, info=info)

            next_state, reward, terminated, truncated, info = env.step(processed_action)
            episode_steps += frame_skip + 1 if use_repeat_action else 1

            episode_reward += reward

            # Crucial: Only treat as 'done' if terminated (failure), not truncated (time limit)
            if (
                use_episode_cutoff
                and episode_steps >= episode_cutoff
                and rewards.numel() >= episode_cutoff
                and rewards[step - episode_cutoff + 1 : step - 1].sum()
                < -episode_cutoff * 0.09
            ):
                # Early termination if no progress for extended period
                penalty = min(
                    cutoff_penalty + (episode_steps - episode_cutoff) * 0.1, 0.0
                )  # Large negative reward for stagnation reduced if car was performing well before
                reward += penalty
                episode_reward += penalty
                truncated = True
            if use_truncated_penalty and truncated:
                reward += truncated_penalty
                episode_reward += truncated_penalty
            if terminated or truncated:
                print(
                    f"Episode {episode} finished - Steps: {episode_steps}, Reward: {episode_reward}"
                )
                if save_recordings and episode_reward > recording_threshold:
                    episode_slice = slice(step - episode_steps + 1, step + 1)
                    episode_recording.push_batch(
                        states[episode_slice].cpu().numpy(),
                        actions[episode_slice].cpu().numpy(),
                    )
                    save_episode_recording(episode, episode_reward, episode_recording)
                    episode_recording.clear()  # Reset for next episode

                next_state, info = env.reset()
                episode += 1
                episode_lengths.append(episode_steps)
                episode_steps = 0
                episode_rewards.append(episode_reward)
                episode_reward = 0

            terminateds[step] = torch.tensor(terminated).to(device)
            truncateds[step] = torch.tensor(truncated).to(device)
            rewards[step] = torch.tensor(reward).to(device)
            state = torch.Tensor(next_state).to(device)
            total_steps += 1
            carryover_reward = episode_reward  # Saves episode reward between rollouts

        # Boostrap value if not done
        with torch.no_grad():
            next_value = agent.policy.get_value(state).reshape(-1)

        returns, advantages = agent.compute_gae(
            rewards, values, terminateds, truncateds, next_value
        )

        rollouts = {
            "states": states,
            "actions": actions,
            "returns": returns,
            "advantages": advantages,
            "values": values,
            "log_probs": log_probs,
        }

        update_metrics = agent.update(rollouts)
        update_count += 1
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        episode_lengths = []  # Clear after logging
        episode_rewards = []

        print(
            f"Update {update_count} completed. Total Steps: {total_steps}. Average Episode Reward: {avg_reward:.2f}"
        )

        log_dict = {
            "policy_loss": update_metrics["policy_loss"],
            "value_function_loss": update_metrics["value_loss"],
            "entropy_loss": update_metrics["entropy_loss"],
            "total_loss": update_metrics["total_loss"],
            "average_episode_reward": avg_reward,
            "average_episode_length": avg_length,
        }

        if WANDB_AVAILABLE and log_wandb:
            wandb.log(log_dict)

        if save_checkpoints and (total_steps >= checkpoint):
            agent.save_model(f"./models/ppo_{episode}_checkpoint.pth")
            checkpoint += checkpoint_interval

    # Save final model
    print(f"\nTraining complete! Saving final model...")
    agent.save_model(f"./models/ppo_{episode}_final.pth")

    if WANDB_AVAILABLE and log_wandb:
        wandb.finish()

    env.close()


def test(device, config, model_path):
    policy_outputs = config["policy_outputs"]
    env = create_env(
        env_name=config.get("env_id", "CarRacing-v3"),
        render_env=True,
        use_grayscale=False,
        use_repeat_action=False,
        use_policy_action_map=True,
        policy_outputs=policy_outputs,
        frame_stack=config.get("frame_stack", 4),
        frame_skip=config.get("frame_skip", 0),
        use_promote_speed=False,
    )

    agent = PPOAgent(device, env, policy_outputs=policy_outputs)
    agent.load_model(model_path)

    state, _ = env.reset()
    state = torch.Tensor(state).to(device)
    steps = 0

    while steps < 5000:
        with torch.no_grad():
            policy_action, _, _ = agent.policy.get_action(state)
        # processed_action = agent.process_action(policy_action)
        next_state, reward, terminated, truncated, info = env.step(policy_action)
        steps += 1
        if terminated or truncated:
            next_state, _ = env.reset()
        state = torch.Tensor(next_state).to(device)


if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    # Ensure replay directory exists
    os.makedirs("./models/replay", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)

    if config.get("evaluation_mode", False):
        print("Running in TEST MODE")
        test(
            device, config, f"./models/{config.get('model_file', 'not_specified.pth')}"
        )
    else:
        print("Running in TRAINING MODE")
        train(device, config)
