import gymnasium as gym
import numpy as np
from agents.sac import SACAgent
import os
import signal
import sys
import time
import yaml
import argparse
from gymnasium.wrappers import RecordVideo
from wrappers import PreprocessCarRacing, RepeatAction, FrameStack


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training will continue without logging")


def train_sac_stepwise(
    env_name="CarRacing-v3",
    max_timesteps=1_000_000,
    start_steps=10000,
    checkpoint_interval=50000,
    use_grayscale=True,
    use_frame_stack=True,
    record_video=False,           # ADD THIS
    video_folder="./videos",      # ADD THIS
    video_interval=10,            # ADD THIS
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

    # Create TRAINING environment (no video recording)
    env = gym.make(env_name, continuous=True)
    
    if use_grayscale and use_frame_stack:
        env = PreprocessCarRacing(env, resize=(84, 84))
        env = RepeatAction(env, skip=4)
        env = FrameStack(env, num_stack=4)
    
    # Create EVALUATION environment (with video recording)
    eval_env = gym.make(env_name, continuous=True, render_mode="rgb_array" if record_video else None)
    
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        eval_env = RecordVideo(
            eval_env, 
            video_folder=video_folder,
            episode_trigger=lambda episode_id: episode_id % video_interval == 0,
            name_prefix="sac-eval",
            disable_logger=True
        )
        print(f"Recording videos to: {video_folder} (every {video_interval} eval episodes)")
    
    if use_grayscale and use_frame_stack:
        eval_env = PreprocessCarRacing(eval_env, resize=(84, 84))
        eval_env = RepeatAction(eval_env, skip=4)
        eval_env = FrameStack(eval_env, num_stack=4)


    print(f"Environment observation shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    
    print(obs_dim)
    # Initialize agent
    agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim)
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
    episode_rewards_history = []
    log_freq = 1000  # Log training progress every N steps
    last_log_step = 0
    episode_steps = 0
    episode_num = 0
    best_eval_score = -np.inf
    
    # Timing statistics
    start_time = time.time()
    last_log_time = start_time
    steps_since_log = 0

    print(f"\nStarting training for {max_timesteps:,} steps...")
    print(f"Warmup period: {start_steps:,} steps\n")

    # fill buffer
    if len(agent.replay_buffer) > 0:
        print(f"[BUFFER] Prefilling buffer ({len(agent.replay_buffer)} steps)...")
        prefill_rewards = []
        prefill_episode_reward = 0
        prefill_episode_count = 0
        log_interval = max(len(agent.replay_buffer) // 10, 1000)  # Log 10 times during prefill
        
        for prefill_step in range(len(agent.replay_buffer)):
            action = env.action_space.sample()
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, term or trunc)
            
            prefill_episode_reward += reward
            obs = next_obs
            
            if term or trunc:
                prefill_rewards.append(prefill_episode_reward)
                prefill_episode_count += 1
                obs, _ = env.reset()
                prefill_episode_reward = 0
            
            # Progress logging
            if (prefill_step + 1) % log_interval == 0:
                progress = 100 * (prefill_step + 1) / len(agent.replay_buffer)
                buffer_size = len(agent.replay_buffer)
                avg_reward = np.mean(prefill_rewards) if prefill_rewards else 0
                print(f"  [{progress:5.1f}%] Step {prefill_step+1:,}/{len(agent.replay_buffer):,} | "
                      f"Buffer: {buffer_size:,} | Episodes: {prefill_episode_count} | "
                      f"Avg Reward: {avg_reward:.2f}")
        
        final_buffer_size = len(agent.replay_buffer)
        final_avg = np.mean(prefill_rewards) if prefill_rewards else 0
        print(f"[OK] Buffer prefilled: {final_buffer_size:,} transitions | "
              f"{prefill_episode_count} episodes | Avg reward: {final_avg:.2f}\n")

    for total_steps in range(1, max_timesteps + 1):
        current_step[0] = total_steps
        
        # Select action: random during warmup, policy after
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(obs, action, reward, next_obs, done)

        # Log actions periodically
        if WANDB_AVAILABLE and total_steps % 100 == 0:
            wandb.log({
                "action/steering": action[0],
                "action/gas": action[1],
                "action/brake": action[2],
            }, step=total_steps)

        # Store transition in the replay buffer
        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        # Training update logic
        buffer_full = len(agent.replay_buffer) >= batch_size
        
        if buffer_full:
            metrics = None
            # for _ in range(gradient_steps):
            metrics = agent.update()
            if metrics and WANDB_AVAILABLE:
                wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=total_steps)

        # Handle episode end
        if done:
            episode_num += 1
            episode_rewards_history.append(episode_reward)
            
            # Keep only recent 100 episodes
            if len(episode_rewards_history) > 100:
                episode_rewards_history.pop(0)
            
            # Determine success for CarRacing
            success = episode_reward >= 900
            
            if WANDB_AVAILABLE:
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_steps,
                    'train/episode_count': episode_num,
                    'train/success': int(success),
                    'train/rolling_mean_reward': np.mean(episode_rewards_history),
                    'train/rolling_std_reward': np.std(episode_rewards_history) if len(episode_rewards_history) > 1 else 0,
                }, step=total_steps)
            
            # Progress logging
            if total_steps - last_log_step >= log_freq:
                progress = 100 * total_steps / max_timesteps
                recent_mean = np.mean(episode_rewards_history[-10:]) if episode_rewards_history else 0
                print(f"[PROGRESS] Step {total_steps:,}/{max_timesteps:,} ({progress:.1f}%) | "
                      f"Episodes: {episode_num} | Recent Avg: {recent_mean:.1f}")
                last_log_step = total_steps
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        
        # Periodic evaluation with video recording
        if total_steps % checkpoint_interval == 0 and total_steps > 0:
            print(f"\n[EVAL] Evaluating at step {total_steps:,}...")
            eval_rewards = []
            
            for eval_ep in range(5):  # 5 evaluation episodes
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_reward = 0
                eval_steps = 0
                
                while not eval_done and eval_steps < 1000:
                    eval_action = agent.select_action(eval_obs, deterministic=True)
                    eval_obs, eval_r, eval_term, eval_trunc, _ = eval_env.step(eval_action)
                    eval_done = eval_term or eval_trunc
                    eval_reward += eval_r
                    eval_steps += 1
                
                eval_rewards.append(eval_reward)
                print(f"    Eval Episode {eval_ep+1}: {eval_reward:.2f}")
            
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            print(f"[EVAL] Mean reward: {eval_mean:.2f} ± {eval_std:.2f}")
            
            if WANDB_AVAILABLE:
                wandb.log({
                    'eval/mean_reward': eval_mean,
                    'eval/std_reward': eval_std,
                    'eval/min_reward': np.min(eval_rewards),
                    'eval/max_reward': np.max(eval_rewards),
                }, step=total_steps)
            
            # Save best model
            if eval_mean > best_eval_score:
                best_eval_score = eval_mean
                agent.save_model(f"./models/sac_best.pth")
                print(f"[BEST] New best model saved! Score: {best_eval_score:.2f}")
            
            # Save checkpoint
            agent.save_model(f"./models/sac_step_{total_steps}.pth")
            print(f"[CHECKPOINT] Saved checkpoint at step {total_steps:,}\n")

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
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    os.makedirs("./models", exist_ok=True)
    
    print("\n--- Training SAC with step-based updates ---")
    
    train_sac_stepwise(
        env_name=config['env_id'],
        max_timesteps=config['max_timesteps'],
        start_steps=config['start_steps'],
        checkpoint_interval=config['checkpoint_interval'],
        use_grayscale=config['use_grayscale'],
        use_frame_stack=config['use_frame_stack'],
        record_video=config.get('record_video', False),
        video_folder=config.get('video_folder', './videos'),
        video_interval=config.get('video_interval', 10)
    )