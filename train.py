import gymnasium as gym
from agents.reinforce import REINFORCEAgent
from agents.vpg import VPGAgent
from agents.ppo import PPOAgent
from agents.sac import SACAgent
import os
import signal
import sys
import numpy as np


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


def train(env_name="CarRacing-v3", algo="vpg", max_episodes=1000):
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
            }
        )
    
    # Create environment
    # Note: render_mode=None for training speed

    env = gym.make(env_name, continuous=True, render_mode=None)
    action_dim = env.action_space.shape[0]
  
    if algo == "reinforce":
        agent = REINFORCEAgent(learning_rate=0.001)
    elif algo == "vpg":
        agent = VPGAgent(learning_rate=0.0003)
    elif algo == "ppo":
        agent = PPOAgent(learning_rate=0.0003, clip_ratio=0.2)
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
    
    random_exploration=True
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)

    total_steps = 0
    UPDATE_EVERY = 50  # Update every 50 steps
    UPDATES_PER_BATCH = 10  # Do 10 updates each time

    
    for episode in range(max_episodes):
        current_episode[0] = episode
        state, _ = env.reset()
        
        # Randomize starting position for SAC
        # if algo == "sac":
        #     position = np.random.randint(len(env.unwrapped.track))
        #     from gymnasium.envs.box2d.car_dynamics import Car
        #     env.unwrapped.car = Car(env.unwrapped.world, *env.unwrapped.track[position][1:4])
        
        episode_reward = 0
        done = False
        steps = 0
        action = env.action_space.sample()  # Initialize action
        loss = None  # Initialize loss to None

        print(f"\n=== Starting Episode {episode} ===")
        print(f"Buffer size: {len(agent.replay_buffer) if algo == 'sac' else 'N/A'}")
        print(f"Total steps so far: {total_steps}")
        
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
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Clip rewards to prevent Q-value explosion
            # if algo == "sac":
            #     reward = np.clip(reward, -10, 10)
            #     # Add off-track penalty
            #     if reward < 0:
            #         reward -= 2.0
            
            # Early termination for poor performance
            if episode_reward < -500:
                done = True
            else:
                done = terminated or truncated
            
            if algo == "sac":
                agent.store_transition(state, action, reward, next_state, log_prob, done)
                
                # # Update once per step after warmup
                # if episode >= start_policy and len(agent.replay_buffer) >= batch_size:
                #     update_result = agent.update()
                #     if update_result is not None:
                #         loss = update_result
            
            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1
            
            if steps > 200 and max_episodes < 10:
                done = True
        
        # if algo != "sac":
        #     print(f"Calling post-episode update for {algo}")
        #     loss = agent.update()
        # elif episode < start_policy:
        #     print(f"Calling post-episode update for SAC warmup")
        #     try:
        #         loss = agent.update()
        #     except Exception as e:
        #         print(f"ERROR during warmup update!")
        #         print(f"Buffer size: {len(agent.replay_buffer)}")
        #         print(f"Error: {e}")
        #         import traceback
        #         traceback.print_exc()
        #         raise
        

        if algo != "sac":
            loss = agent.update()
        else:
            # SAC updates once per episode
            loss = agent.update()

        # Log metrics to WandB if available
        log_dict = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_steps": steps,
        }
        
        # Handle tuple return from SAC update
        if algo == "sac" and loss is not None and isinstance(loss, tuple):
            policy_loss, q_loss, info = loss
            log_dict.update({"policy_loss": policy_loss, "q_loss": q_loss, "alpha": info.get("alpha", None)})
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Policy Loss: {policy_loss:.4f} | Q Loss: {q_loss:.4f} | Alpha: {info.get('alpha', None)}")
            if episode % 5 == 0:
                print(f"Action stats - Steering: {action[0]:.2f}, Gas: {action[1]:.2f}, Brake: {action[2]:.2f}")
        elif loss is not None and isinstance(loss, tuple):
            policy_loss, vf_loss, info = loss
            log_dict.update({"policy_loss": policy_loss, "value_function_loss": vf_loss})
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Policy Loss: {policy_loss:.4f} | VF Loss: {vf_loss:.4f}")
        else:
            if loss is not None:
                log_dict["loss"] = loss
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Loss: {loss:.4f}")
            else:
                print(f"Episode {episode} | Reward: {episode_reward:.2f} | Loss: None (buffer warming up)")
        
        if episode % 10 == 0:
            agent.save_model(f"./models/{algo}_{episode}.pth")
    
    # Save final model
    print(f"\nTraining complete! Saving final model...")
    agent.save_model(f"./models/{algo}_{max_episodes-1}_final.pth")
    
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
    # train(algo="reinforce", max_episodes=5000)

    # print("--- Testing VPG for 3000 episodes ---")
    #train(algo="vpg", max_episodes=3000)
        
    # print("\n--- Testing PPO for 3000 episodes ---")
    # train(algo="reinforce", max_episodes=200)

    print("\n--- Testing SAC for 1000 episodes ---")
    train(algo="sac", max_episodes=1000)