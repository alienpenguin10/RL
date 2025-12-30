import numpy as np
import torch
import gymnasium as gym
from agents.ppo import PPOAgent
from agents.recording_buffer import RecordingBuffer
from CarRacingEnv.env_wrapper import ProcessedFrame, FrameStack
import signal
import sys
import os
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

""" Hyperparameters """
TEST_MODE = False
MODEL_FILE = "ppo_1_final.pth"  # Replace with your model file for testing
RENDER_ENV = False
LOG_WANDB = True
SAVE_CHECKPOINTS = False
TOTAL_TIMESTEPS = 100000

HORIZON = 2048
NUM_UPDATES = int(TOTAL_TIMESTEPS / HORIZON) # 100000 / 2048 = 244
NUM_EPOCHS = 4
NUM_MINIBATCHES = 8
BATCH_SIZE = HORIZON // NUM_MINIBATCHES # 2048 // 8 = 256
FRAME_STACKING = True
NUM_FRAMES = 6
SKIP_FRAMES = 4
CHECKPOINT_INTERVAL = HORIZON

EPISODE_CUTOFF = 300  # Early termination if no progress for n steps
CUTOFF_PENALTY = -100.0  # Penalty for early cutoff
TRUNCATED_PENALTY = -20.0  # Penalty for episode truncation due to time limit
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILONS = 0.1 # Clipping ratio for PPO
VALUE_COEFF = 0.01
ENTROPY_COEFF = 0.02
ENTROPY_DECAY = 1.0 # Set to <1.0 to decay entropy coefficient over time
L2_REG = 1e-2 # Set to 0.0 to disable L2 regularization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env_name='CarRacing-v3', render_env=False, log_wandb=False):
    # Initialize WandB if available
    if WANDB_AVAILABLE and log_wandb:
        wandb.init(
            project="rl-training",
            name=f"ppo_{env_name}",
            config={
                "algorithm": "ppo",
                "environment": env_name,
                "max_timesteps": TOTAL_TIMESTEPS,
                "buffer_size": HORIZON,
                "mini_batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
            }
        )
    
    env = gym.make(f'{env_name}', render_mode='human' if render_env else None)
    env = ProcessedFrame(env)
    env = FrameStack(env, num_frames=NUM_FRAMES, skip_frames=SKIP_FRAMES)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = PPOAgent(DEVICE, env, NUM_EPOCHS, BATCH_SIZE,
                     GAMMA, GAE_LAMBDA, VALUE_COEFF, EPSILONS,
                     LEARNING_RATE, NUM_UPDATES, ENTROPY_COEFF, ENTROPY_DECAY,
                     L2_REG)

    state, _ = env.reset()
    state = torch.Tensor(state).to(DEVICE)
    total_steps = 0
    update_count = 0
    episode = 0 # Track current episode for saving on interrupt
    checkpoint = 0
    episode_steps = 0
    episode_reward = 0
    episode_rewards = []
    carryover_reward = 0
    episode_recording = RecordingBuffer(capacity=1000) # episode is max 1000 steps

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
        states, actions = recording_buffer.get_recording()
        np.savez(replay_path, states=states, actions=actions)
        print(f"Replay saved to {replay_path}")

    rolling_speed = 0.0
    # steering_buffer = deque([], maxlen=5)
    
    while total_steps < TOTAL_TIMESTEPS:
        states = torch.zeros((HORIZON, *env.observation_space.shape)).to(DEVICE)
        actions = torch.zeros((HORIZON, env.action_space.shape[0])).to(DEVICE)
        # actions = torch.zeros((HORIZON, 2)).to(DEVICE)  # Only steer and speed
        rewards = torch.zeros((HORIZON)).to(DEVICE)
        terminateds = torch.zeros((HORIZON)).to(DEVICE)
        truncateds = torch.zeros((HORIZON)).to(DEVICE)
        values = torch.zeros((HORIZON)).to(DEVICE)
        log_probs = torch.zeros((HORIZON)).to(DEVICE)

        # state, _ = env.reset()
        # state = torch.Tensor(state).to(DEVICE)
        episode_reward += carryover_reward

        # Rollout
        for step in range(HORIZON):
            states[step] = state
            with torch.no_grad():
                raw_action, log_prob, value = agent.policy.get_action(state)
                values[step] = torch.tensor(value).to(DEVICE)       
            actions[step] = torch.tensor(raw_action).to(DEVICE)
            log_probs[step] = torch.tensor(log_prob).to(DEVICE)
            processed_action = agent.process_action(raw_action, rolling_speed=rolling_speed)
            rolling_speed = 0.9999*rolling_speed + processed_action[1]*0.1 - processed_action[2]
            # steering_buffer.append(processed_action[0])

            next_state, reward, terminated, truncated, info = env.step(processed_action)
            episode_steps += 1

            episode_reward += reward

            # Crucial: Only treat as 'done' if terminated (failure), not truncated (time limit)
            if (episode_steps >= EPISODE_CUTOFF and rewards.numel() >= EPISODE_CUTOFF and rewards[step-EPISODE_CUTOFF+1:step-1].sum() < -EPISODE_CUTOFF*0.09):
                # Early termination if no progress for extended period
                penalty = min(CUTOFF_PENALTY + (episode_steps-EPISODE_CUTOFF) * 0.1, 0.0) # Large negative reward for stagnation reduced if car was performing well before
                reward += penalty
                episode_reward += penalty
                truncated = True
            if truncated:
                reward += TRUNCATED_PENALTY
                episode_reward += TRUNCATED_PENALTY
            if terminated or truncated:
                print(f"Episode {episode} finished - Steps: {episode_steps}, Reward: {episode_reward}")
                if episode_reward > 800.0:
                    episode_slice = slice(step-episode_steps+1, step)
                    episode_recording.push_batch(states[episode_slice].cpu().numpy(), 
                                                actions[episode_slice].cpu().numpy())
                    save_episode_recording(episode, episode_reward, episode_recording)
                    episode_recording.clear() # Reset for next episode
                
                next_state, _ = env.reset()
                episode += 1
                episode_steps = 0
                episode_rewards.append(episode_reward)
                episode_reward = 0
                rolling_speed = 0.0
                # steering_buffer.clear()

            terminateds[step] = torch.tensor(terminated).to(DEVICE)
            truncateds[step] = torch.tensor(truncated).to(DEVICE)
            rewards[step] = torch.tensor(reward).to(DEVICE)
            state = torch.Tensor(next_state).to(DEVICE)
            total_steps += 1
            carryover_reward = episode_reward # Saves episode reward between rollouts
        
        # Boostrap value if not done
        with torch.no_grad():
            next_value = agent.policy.get_value(state).reshape(-1)

        returns, advantages = agent.compute_gae(rewards, values, terminateds, truncateds, next_value)
        
        rollouts = {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'values': values,
            'log_probs': log_probs,
        }

        update_metrics = agent.update(rollouts)
        update_count += 1
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        episode_rewards = []  # Clear after logging

        print(f"Update {update_count} completed. Total Steps: {total_steps}. Average Episode Reward: {avg_reward:.2f}")

        log_dict = {
            "policy_loss": update_metrics['policy_loss'],
            "value_function_loss": update_metrics['value_loss'],
            "entropy_loss": update_metrics['entropy_loss'],
            "total_loss": update_metrics['total_loss'],
            "average_episode_reward": avg_reward,
        }

        if WANDB_AVAILABLE and log_wandb:
            wandb.log(log_dict)

        
        if SAVE_CHECKPOINTS and (total_steps >= checkpoint):
            agent.save_model(f"./models/ppo_{episode}_checkpoint.pth")
            checkpoint += CHECKPOINT_INTERVAL
    
    # Save final model
    print(f"\nTraining complete! Saving final model...")
    agent.save_model(f"./models/ppo_{episode}_final.pth")

    if WANDB_AVAILABLE:
        wandb.finish()
    
    env.close()

def test(model_path="./models/ppo_final.pth"):
    env = gym.make(f'{env_name}', render_mode='human' if render_env else None)
    env = ProcessedFrame(env)
    env = FrameStack(env, num_frames=NUM_FRAMES, skip_frames=SKIP_FRAMES)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = PPOAgent(env)
    agent.load_model(model_path)

    state, _ = env.reset()
    state = torch.Tensor(state).to(DEVICE)
    steps = 0

    while steps < TOTAL_TIMESTEPS:
        with torch.no_grad():
            action, _, _ = agent.policy.get_action(state)
        processed_action = agent.process_action(action)
        next_state, reward, terminated, truncated, info = env.step(processed_action)
        steps += 1
        if terminated or truncated:
            next_state, _ = env.reset()
        state = torch.Tensor(next_state).to(DEVICE)

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)

    # Ensure replay directory exists
    os.makedirs("./models/replay", exist_ok=True)

    env_name = 'CarRacing-v3'
    log_wandb = LOG_WANDB
    render_env = RENDER_ENV
    if TEST_MODE:
        print("Running in TEST MODE")
        test(f"./models/{MODEL_FILE}")
    else:
        print("Running in TRAINING MODE")
        train(env_name=env_name, render_env=render_env, log_wandb=log_wandb)