import gymnasium as gym
import numpy as np
from agents.sac import SACAgent
import os
import signal
import sys
import time
import cv2

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training will continue without logging")


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB observation to grayscale"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
    
    def observation(self, obs):
        # Convert to grayscale and resize to 84x84
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


class FrameStackWrapper(gym.Wrapper):
    """Stack last n frames"""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = []
        
        obs_shape = env.observation_space.shape
        # Stack in channels-first format: (n_stack, H, W)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(n_stack, *obs_shape),  # (4, 84, 84)
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.n_stack
        return self._get_observation(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        # Stack along axis 0 to get (4, 84, 84) instead of (84, 84, 4)
        return np.stack(self.frames, axis=0)  # Stack as first dimension


def train_sac_stepwise(
    env_name="CarRacing-v3",
    max_timesteps=1_000_000,
    start_steps=10000,
    checkpoint_interval=50000,
    use_grayscale=True,
    use_frame_stack=True,
):
    # Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name=f"sac-stepwise-{env_name}",
            config={
                "algorithm": "sac",
                "environment": env_name,
                "max_timesteps": max_timesteps,
                "start_steps": start_steps,
                "grayscale": use_grayscale,
                "frame_stack": use_frame_stack,
            }
        )

    # Create environment with custom wrappers
    env = gym.make(env_name, continuous=True)
    
    if use_grayscale:
        env = GrayscaleWrapper(env)
    
    if use_frame_stack:
        env = FrameStackWrapper(env, n_stack=4)
    
    print(f"Environment observation shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = SACAgent(action_dim=action_dim)
    batch_size = agent.batch_size
    
    print(f"Training with batch size: {batch_size}")
    print(f"Device: {agent.device}")

    # Track current step for graceful shutdown
    current_step = [0]
    
    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model at step {current_step[0]}...")
        agent.save_model(f"./models/sac_step_{current_step[0]}_interrupted.pth")
        print(f"Model saved!")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, save_on_interrupt)

    # Initialize episode tracking
    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    
    # Timing statistics
    start_time = time.time()
    last_log_time = start_time
    steps_since_log = 0

    print(f"\nStarting training for {max_timesteps:,} steps...")
    print(f"Warmup period: {start_steps:,} steps\n")

    for total_steps in range(1, max_timesteps + 1):
        current_step[0] = total_steps
        
        # Select action: random during warmup, policy after
        if total_steps < start_steps:
            action = env.action_space.sample()
        else:
            action, log_prob = agent.select_action(obs)

            # Log actions periodically
            if WANDB_AVAILABLE and total_steps % 100 == 0:
                wandb.log({
                    "action/steering": action[0],
                    "action/gas": action[1],
                    "action/brake": action[2],
                    "action/log_prob": log_prob.item() if log_prob is not None else 0,
                }, step=total_steps)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition in the replay buf
        agent.store_transition(obs, action, reward, next_obs, done)

        # Update every step after warmup
        if total_steps >= start_steps and len(agent.replay_buffer) >= batch_size and total_steps % 4 == 0:
            update_result = agent.update()
            
            if update_result is not None and WANDB_AVAILABLE and total_steps % 100 == 0:
                wandb.log({
                    "loss/critic": update_result['critic_loss'],
                    "loss/actor": update_result['actor_loss'],
                    "loss/alpha": update_result['alpha_loss'],
                    "alpha": update_result['alpha'],
                    "entropy": update_result['entropy'],
                    "q1_mean": update_result['q1_mean'],
                    "q2_mean": update_result['q2_mean'],
                    "buffer_size": len(agent.replay_buffer),
                }, step=total_steps)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        steps_since_log += 1

        # Handle episode end
        if done:
            if WANDB_AVAILABLE:
                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/steps": episode_steps,
                    "episode/num": episode_num,
                    "episode/avg_reward_per_step": episode_reward / episode_steps if episode_steps > 0 else 0,
                }, step=total_steps)
            
            # Print progress every episode
            current_time = time.time()
            time_elapsed = current_time - last_log_time
            steps_per_sec = steps_since_log / time_elapsed if time_elapsed > 0 else 0
            
            print(f"Episode {episode_num:4d} | "
                  f"Step {total_steps:7d}/{max_timesteps:7d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"EpLen: {episode_steps:4d} | "
                  f"SPS: {steps_per_sec:.1f}")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_num += 1
            
            last_log_time = current_time
            steps_since_log = 0

        # Save checkpoint every checkpoint_interval steps
        if total_steps % checkpoint_interval == 0:
            agent.save_model(f"./models/sac_step_{total_steps}.pth")
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Checkpoint saved at step {total_steps:,}")
            print(f"Time elapsed: {elapsed/3600:.2f} hours")
            print(f"Progress: {total_steps/max_timesteps*100:.1f}%")
            print(f"{'='*60}\n")

    # Save final model
    agent.save_model(f"./models/sac_final.pth")
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("Training complete! Final model saved.")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total episodes: {episode_num}")
    print(f"Average time per 1000 steps: {total_time/(max_timesteps/1000):.2f}s")
    print(f"{'='*60}")

    if WANDB_AVAILABLE:
        wandb.finish()
    env.close()


if __name__ == "__main__":
    if WANDB_AVAILABLE:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            print("Warning: WANDB_API_KEY not set. Continuing without WandB logging.")
    
    os.makedirs("./models", exist_ok=True)
    
    print("\n--- Training SAC with step-based updates (custom wrappers) ---")
    
    train_sac_stepwise(
        max_timesteps=500_000,
        start_steps=5000,
        checkpoint_interval=25000,
        use_grayscale=True,
        use_frame_stack=True
    )