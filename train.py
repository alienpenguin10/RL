import gymnasium as gym
from agents.reinforce import REINFORCEAgent
from agents.vpg import VPGAgent
from agents.ppo import PPOAgent
from agents.sac import SACAgent
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
  
    if algo == "reinforce":
        agent = REINFORCEAgent(learning_rate=0.001)
    elif algo == "vpg":
        agent = VPGAgent(learning_rate=0.0003)
    elif algo == "ppo":
        agent = PPOAgent(learning_rate=0.0003, clip_ratio=0.2)
    elif algo == "sac":
        agent = SACAgent()
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
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)

    for episode in range(max_episodes):
        current_episode[0] = episode
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select action
            if algo == "reinforce":
                action, log_prob = agent.select_action(state)
                value = None
            else: # VPG, PPO same
                action, log_prob, value = agent.select_action(state)
            
            # Step env
            # Note: gymnasium returns terminated, truncated
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            # Ensure we pass all required arguments
            agent.store_transition(state, action, reward, next_state, log_prob, done, value)
            
            state = next_state
            episode_reward += reward
            steps += 1
            # Safety break for very long episodes during testing
            if steps > 200 and max_episodes < 10:
                done = True
            
        # Update at end of episode (REINFORCE) or end of batch (VPG - here treating episode as batch)
        loss = agent.update()
        
        # Log metrics to WandB if available
        log_dict = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_steps": steps,
        }
        
        # Handle tuple return from VPG update
        if isinstance(loss, tuple):
            policy_loss, vf_loss, info = loss
            log_dict.update({"policy_loss": policy_loss,"value_function_loss": vf_loss,})
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Policy Loss: {policy_loss:.4f} | VF Loss: {vf_loss:.4f}")
        else:
            log_dict["loss"] = loss
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Loss: {loss:.4f}")
        
        if WANDB_AVAILABLE:
            wandb.log(log_dict)
        
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

    print("\n--- Testing SAC for 100 episodes ---")
    train(algo="sac", max_episodes=100)