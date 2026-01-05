import gymnasium as gym
from agents.reinforce import REINFORCEAgent
from agents.vpg import VPGAgent
from agents.ppo_old import PPOAgent
from agents.sac import SACAgent
import os
import signal
import sys
import numpy as np
import time
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation

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


def train(env_name="CarRacing-v3", algo="vpg", max_episodes=1000, 
          update_frequency=1, updates_per_step=1):
    """
    Args:
        update_frequency: Update every N steps (e.g., 4 means update every 4 steps)
        updates_per_step: How many gradient updates to perform when updating (default 1)
    """
    # Initialize WandB if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name=f"{algo}-{env_name}",
            config={
                "algorithm": algo,
                "environment": env_name,
                "max_episodes": max_episodes,
                "learning_rate": 0.001 if algo == "reinforce" else 0.0003,
                "update_frequency": update_frequency,
                "updates_per_step": updates_per_step,
            }
        )
    
    # Create environment
    env = gym.make(env_name, continuous=True, render_mode=None)
    env = GrayscaleObservation(env, keep_dim=False)  # RGB -> Grayscale
    env = FrameStackObservation(env, 4)
    action_dim = env.action_space.shape[0]
  
    if algo == "reinforce":
        agent = REINFORCEAgent(learning_rate=0.001)
    elif algo == "vpg":
        agent = VPGAgent(learning_rate=0.0003)
    elif algo == "ppo":
        agent = PPOAgent(learning_rate=0.0003)
    elif algo == "sac":
        agent = SACAgent(action_dim=action_dim)
        batch_size = agent.batch_size
        buffer_path = "./models/sac_replay_buffer.pkl"
        buffer_loaded = False
        if os.path.exists(buffer_path):
            agent.load_replay_buffer(buffer_path)
            print("Loaded replay buffer from disk.")
            buffer_loaded = True
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
        
    print(f"Training {algo} on {env_name} using device: {agent.device}")
    print(f"Update frequency: every {update_frequency} steps")
    print(f"Updates per step: {updates_per_step}")
    
    # Track current episode for saving on interrupt
    current_episode = [0]
    
    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model from episode {current_episode[0]}...")
        agent.save_model(f"./models/{algo}_{current_episode[0]}_interrupted.pth")
        print(f"Model saved to ./models/{algo}_{current_episode[0]}_interrupted.pth")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)

    def generate_action(prev_action):
        if np.random.randint(3) % 3:
            return prev_action
        index = np.random.randn(3)
        index[1] = np.abs(index[1])
        index = np.argmax(index)
        mask = np.zeros(3)
        mask[index] = 1
        action = np.random.randn(3)
        action = np.tanh(action)
        action[1] = (action[1] + 1) / 2
        action[2] = (action[2] + 1) / 2
        return action * mask
    
    start_policy = 150
    if algo == "sac" and buffer_loaded:
        start_policy = 0
    
    random_exploration = True
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)

    total_steps = 0
    
    # SAC-specific tracking for logging
    total_updates = 0
    running_policy_loss = []
    running_q_loss = []
    running_alpha = []
    
    # Timing statistics
    episode_times = []
    update_times = []
    env_step_times = []
    
    for episode in range(max_episodes):
        episode_start_time = time.time()
        current_episode[0] = episode
        state, _ = env.reset()
        
        # Randomize starting position for SAC
        if algo == "sac":
            position = np.random.randint(len(env.unwrapped.track))
            from gymnasium.envs.box2d.car_dynamics import Car
            env.unwrapped.car = Car(env.unwrapped.world, *env.unwrapped.track[position][1:4])
        
        episode_reward = 0
        done = False
        steps = 0
        action = env.action_space.sample()

        print(f"\n=== Episode {episode} ===")
        if episode % 10 == 0:  # Less verbose logging
            print(f"Buffer size: {len(agent.replay_buffer) if algo == 'sac' else 'N/A'}")
            print(f"Total steps: {total_steps}, Total updates: {total_updates if algo == 'sac' else 'N/A'}")
            if episode_times:
                print(f"Avg episode time: {np.mean(episode_times[-10:]):.2f}s")
        
        # Episode-specific tracking for SAC
        episode_policy_losses = []
        episode_q_losses = []
        episode_alphas = []
        episode_update_time = 0
        episode_env_time = 0
        
        while not done:
            # Select action based on algorithm
            if algo == "reinforce":
                action, log_prob = agent.select_action(state)
                value = None
            
            elif algo == "sac":
                # Save buffer after warmup (only once)
                if episode == start_policy and not buffer_loaded:
                    print("Saving replay buffer, size:", len(agent.replay_buffer))
                    agent.save_replay_buffer("./models/sac_replay_buffer.pkl")
                    print("Replay buffer saved after warmup.")
                
                # Use random exploration during warmup, agent policy after
                if episode < start_policy:
                    action = generate_action(action) if random_exploration else env.action_space.sample()
                    log_prob = None
                else:
                    action, log_prob = agent.select_action(state)
            
            else:  # VPG, PPO
                action, log_prob, value = agent.select_action(state)
            
            # Step environment (time this)
            env_start = time.time()
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_env_time += (time.time() - env_start)
            
            # Early termination for poor performance
            if episode_reward < -500:
                done = True
            else:
                done = terminated or truncated
            
            # Store transition and update for SAC
            if algo == "sac":
                agent.store_transition(state, action, reward, next_state, log_prob, done)
                
                # ===== OPTIMIZED: Update every N steps with M gradient updates =====
                should_update = (episode >= start_policy and 
                               len(agent.replay_buffer) >= batch_size and
                               steps % update_frequency == 0)
                
                if should_update:
                    update_start = time.time()
                    
                    # Perform multiple gradient updates
                    for _ in range(updates_per_step):
                        update_result = agent.update()
                        
                        if update_result is not None:
                            policy_loss, q_loss, info = update_result
                            episode_policy_losses.append(policy_loss)
                            episode_q_losses.append(q_loss)
                            episode_alphas.append(info.get("alpha", 0))
                            total_updates += 1
                    
                    episode_update_time += (time.time() - update_start)
            
            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1
            
            if steps > 200 and max_episodes < 10:
                done = True
        
        # ===== Post-episode updates for on-policy algorithms =====
        if algo != "sac":
            # On-policy algorithms: update once per episode with collected data
            loss = agent.update()
        else:
            # SAC: Already updated during episode, just compute averages for logging
            if len(episode_policy_losses) > 0:
                avg_policy_loss = np.mean(episode_policy_losses)
                avg_q_loss = np.mean(episode_q_losses)
                avg_alpha = np.mean(episode_alphas)
                loss = (avg_policy_loss, avg_q_loss, {"alpha": avg_alpha})
                
                # Update running statistics
                running_policy_loss.extend(episode_policy_losses)
                running_q_loss.extend(episode_q_losses)
                running_alpha.extend(episode_alphas)
                
                # Keep only last 100 updates for running average
                if len(running_policy_loss) > 100:
                    running_policy_loss = running_policy_loss[-100:]
                    running_q_loss = running_q_loss[-100:]
                    running_alpha = running_alpha[-100:]
            else:
                loss = None
        
        # Episode timing
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        if len(episode_times) > 100:
            episode_times = episode_times[-100:]
        
        if episode_update_time > 0:
            update_times.append(episode_update_time)
        if episode_env_time > 0:
            env_step_times.append(episode_env_time)

        # Log metrics to WandB if available
        log_dict = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_steps": steps,
            "episode_time": episode_time,
        }
        
        # Handle tuple return from SAC update
        if algo == "sac" and loss is not None and isinstance(loss, tuple):
            policy_loss, q_loss, info = loss
            log_dict.update({
                "policy_loss": policy_loss, 
                "q_loss": q_loss, 
                "alpha": info.get("alpha", None),
                "updates_this_episode": len(episode_policy_losses),
                "total_updates": total_updates,
                "running_avg_policy_loss": np.mean(running_policy_loss) if running_policy_loss else 0,
                "running_avg_q_loss": np.mean(running_q_loss) if running_q_loss else 0,
                "update_time": episode_update_time,
                "env_time": episode_env_time,
                "update_time_fraction": episode_update_time / episode_time if episode_time > 0 else 0,
            })
            
            print(f"Ep {episode} | Reward: {episode_reward:.1f} | Steps: {steps} | "
                  f"Updates: {len(episode_policy_losses)} | Time: {episode_time:.1f}s")
            
            if episode % 5 == 0:
                print(f"  Losses - Policy: {policy_loss:.4f}, Q: {q_loss:.4f}, Alpha: {info.get('alpha', 0):.4f}")
                print(f"  Timing - Env: {episode_env_time:.1f}s ({episode_env_time/episode_time*100:.0f}%), "
                      f"Update: {episode_update_time:.1f}s ({episode_update_time/episode_time*100:.0f}%)")
            
            # Print running averages every 20 episodes
            if episode % 20 == 0 and episode > 0 and running_policy_loss:
                print(f"  Running Avg (last 100 updates) - "
                      f"Policy: {np.mean(running_policy_loss):.4f}, "
                      f"Q: {np.mean(running_q_loss):.4f}")
                print(f"  Avg time/episode (last 10): {np.mean(episode_times[-10:]):.1f}s")
                
        elif loss is not None and isinstance(loss, tuple):
            policy_loss, vf_loss, info = loss
            log_dict.update({"policy_loss": policy_loss, "value_function_loss": vf_loss})
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Time: {episode_time:.1f}s")
        else:
            if loss is not None:
                log_dict["loss"] = loss
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Time: {episode_time:.1f}s")
            else:
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Time: {episode_time:.1f}s (warmup)")
        
        if WANDB_AVAILABLE:
            wandb.log(log_dict)
        
        # Save less frequently to reduce I/O overhead
        if episode % 50 == 0 and episode > 0:
            agent.save_model(f"./models/{algo}_{episode}.pth")
    
    # Save final model
    print(f"\nTraining complete! Saving final model...")
    agent.save_model(f"./models/{algo}_{max_episodes-1}_final.pth")
    
    if algo == "sac":
        print(f"\n=== SAC Training Statistics ===")
        print(f"Total gradient updates: {total_updates}")
        print(f"Total environment steps: {total_steps}")
        print(f"Updates per step ratio: {total_updates / total_steps if total_steps > 0 else 0:.2f}")
        print(f"Average episode time: {np.mean(episode_times):.2f}s")
        if update_times:
            print(f"Average update time per episode: {np.mean(update_times):.2f}s")
        if env_step_times:
            print(f"Average env step time per episode: {np.mean(env_step_times):.2f}s")
        if running_policy_loss:
            print(f"Final running avg policy loss: {np.mean(running_policy_loss):.4f}")
            print(f"Final running avg Q loss: {np.mean(running_q_loss):.4f}")
            print(f"Final alpha: {running_alpha[-1]:.4f}")
    
    if WANDB_AVAILABLE:
        wandb.finish()
    
    env.close()


if __name__ == "__main__":
    # Check for WandB API key
    if WANDB_AVAILABLE:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            print("Warning: WANDB_API_KEY not set. Continuing without WandB logging.")
    
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    # OPTIMIZATION OPTIONS - Choose one:
    
    # Option 1: Update every 4 steps (4x faster, still good performance)
    print("\n--- Training SAC with optimized updates (every 4 steps) ---")
    train(algo="sac", max_episodes=1000, update_frequency=4, updates_per_step=1)
    
    # Option 2: Update every 10 steps (10x faster, may impact performance slightly)
    # train(algo="sac", max_episodes=1000, update_frequency=10, updates_per_step=1)
    
    # Option 3: Update every 4 steps with 2 gradient updates each time (2x updates at 4x spacing)
    # train(algo="sac", max_episodes=1000, update_frequency=4, updates_per_step=2)
    
    # Option 4: Standard approach (update every step) - slowest but most stable
    # train(algo="sac", max_episodes=1000, update_frequency=1, updates_per_step=1)