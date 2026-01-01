from CarRacingEnv.env_wrapper import ActionRemapWrapper, FrameStack, ProcessedFrame
import gymnasium as gym
import numpy as np
from agents.reinforce import REINFORCEAgent
from agents.vpg import VPGAgent
from agents.ppo_old import PPOAgent
import torch
import os
import signal
import sys

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


def train(env_name="CarRacing-v3", algo="vpg", max_train_iters=1000):
    # Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name=f"{algo}-{env_name}",
            config={
                "algorithm": algo,
                "environment": env_name,
                "max_episodes": max_train_iters,
                "learning_rate": 0.001 if algo == "reinforce" else 0.0003,
            }
        )
    
    # Create environment
    # Note: render_mode=None for training speed

    env = gym.make(env_name, continuous=True, render_mode=None)
    env = ProcessedFrame(env)
    env = ActionRemapWrapper(env)
    env = FrameStack(env, num_frames=4, skip_frames=2)
  
    # Buffer size: how many steps to collect before each update
    buffer_size = 2048  # Standard PPO uses 2048
    batch_size = 64   # Mini-batch size for updates (smaller batch sizes often generalise better)
    agent = PPOAgent(state_dim=(4, 84, 96), learning_rate=0.0003, clip_ratio=0.2, mini_batch_size=batch_size, buffer_size=buffer_size)
    
        
    print(f"Training {algo} on {env_name} using device: {agent.device}")
    
    # Track current episode for saving on interrupt
    current_episode = [0]  # Use list to allow modification in nested function
    
    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model from episode {current_episode[0]}...")
        agent.save_model(f"./models/{algo}_{current_episode[0]}_interrupted.pth")
        print(f"Model saved to ./models/{algo}_{current_episode[0]}_interrupted.pth")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)
    state, _ = env.reset()
    steps = 0
    max_steps = max_train_iters

    while steps < max_steps:
        episode = 0
        episode_steps = 0
        episode_reward = 0
        steps_in_episode = []
        episode_rewards = []

        for t in range(buffer_size):
            action, log_prob, val = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, log_prob, done, val)
            state = next_state
            steps += 1
            episode_steps += 1
            episode_reward += reward

            # Handle episode termination
            if done:
                # Finish path with value=0 since it's terminal
                agent.finish_path(last_val=0)
                episode += 1
                current_episode[0] = episode

                # Log episode reward
                steps_in_episode.append(episode_steps)
                episode_rewards.append(episode_reward)

                # Reset for next episode
                episode_steps = 0
                episode_reward = 0
                state, _ = env.reset()
            elif t == buffer_size - 1:
                # Buffer is full but episode not done: Bootstrap!
                # We need the value of the current 'state' (which is actually next_state of the last step)
                _, _, last_val = agent.select_action(state)
                agent.finish_path(last_val=last_val)


        # 2. Update after buffer is full
        loss = agent.update()
        policy_loss, vf_loss = loss

        avg_episode_steps = np.mean(steps_in_episode) if len(steps_in_episode) > 0 else 0
        avg_episode_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0

        log_dict = {
            "policy_loss": policy_loss,
            "value_function_loss": vf_loss,
            "total_steps": steps,
            "average_episode_steps": avg_episode_steps,
            "average_episode_reward": avg_episode_reward,
            "episode": episode,
        }
        # print(f"Update at step {steps} | Policy Loss: {policy_loss:.4f} | VF Loss: {vf_loss:.4f}")

        if WANDB_AVAILABLE:
            wandb.log(log_dict)
        
      
    
    # Save final model
    print(f"\nTraining complete! Saving final model...")
    agent.save_model(f"./models/{algo}_{max_train_iters-1}_final.pth")
    
    if WANDB_AVAILABLE:
        wandb.finish()
    
    env.close()

if __name__ == "__main__":
    # Check for WandB API key - fail if not set
    if WANDB_AVAILABLE:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY is not set in the environment variables. Please create a .env file with your WandB API key. See README.md for setup instructions.")
    
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    # print("\n--- Testing REINFORCE for 5000 episodes ---")
    #train(algo="reinforce", max_episodes=5000)

    # print("--- Testing VPG for 3000 episodes ---")
    #train(algo="vpg", max_episodes=3000)
        
    # print("\n--- Testing PPO for 3000 episodes ---")
    train(algo="ppo", max_train_iters=1000000)
