import gymnasium as gym
import numpy as np
import torch
import os
import json
import argparse
from dotenv import load_dotenv

# Import agents
from agents.reinforce import REINFORCEAgent
from agents.vpg import VPGAgent
from agents.ppo import PPOAgent

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be skipped.")

def load_agent(algo, model_path, device):
    """Initializes the agent and loads the state dict."""
    print(f"Loading {algo} agent from {model_path}...")
    
    # We initialize with default hyperparameters as they don't affect inference
    # Only the network architecture matters for loading weights
    if algo == "reinforce":
        agent = REINFORCEAgent()
    elif algo == "vpg":
        agent = VPGAgent()
    elif algo == "ppo":
        agent = PPOAgent()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Load model
    agent.load_model(model_path)
    agent.policy_network.eval() # Set to eval mode (affects dropout/batchnorm if present)
    return agent

def evaluate(algo, model_path, env_name="CarRacing-v3", episodes=10, render=False, use_wandb=False):
    # Load environment variables
    load_dotenv()

    # WandB Setup
    if use_wandb and WANDB_AVAILABLE:
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        
        wandb.init(
            project="rl-evaluation",
            name=f"eval-{algo}-{os.path.basename(model_path)}",
            config={
                "algorithm": algo,
                "model_path": model_path,
                "episodes": episodes
            }
        )

    # Environment Setup
    render_mode = "human" if render else None
    try:
        env = gym.make(env_name, continuous=True, render_mode=render_mode)
    except gym.error.Error as e:
        print(f"Error creating environment: {e}")
        return

    # Agent Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        agent = load_agent(algo, model_path, device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"--- Starting Evaluation on {device} ---")
    
    episode_rewards = []
    episode_lengths = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done:
            # Preprocess state using the agent's internal helper
            state_tensor = agent.preprocess_state(state)
            
            # Use the policy network's act method for inference (no_grad is built-in there)
            action = agent.policy_network.act(state_tensor)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            # Safety break for stuck agents
            if steps > 2000:
                print(f"Episode {ep+1} truncated due to length.")
                done = True

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Steps: {steps}")

    env.close()

    # Calculate Statistics
    results = {
        "algorithm": algo,
        "model_name": os.path.basename(model_path),
        "episodes_run": episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "raw_rewards": episode_rewards
    }

    # Console Summary
    print("\n--- Evaluation Summary ---")
    print(f"Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Max Reward:  {results['max_reward']:.2f}")

    # Log to WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(results)
        wandb.finish()

    # Save to JSON
    output_filename = f"eval_results_{algo}_{os.path.basename(model_path)}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filename}")

def get_latest_model(algo, model_dir="./models"):
    """Finds the model file with the highest episode number for the given algo."""
    if not os.path.exists(model_dir):
        return None
        
    files = [f for f in os.listdir(model_dir) if f.startswith(algo) and f.endswith(".pth")]
    if not files:
        return None

    # Sort by episode number (assuming format algo_123.pth)
    def extract_ep(f):
        try:
            return int(f.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            return -1
            
    files.sort(key=extract_ep)
    return os.path.join(model_dir, files[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Agents")
    parser.add_argument("--algo", type=str, required=True, choices=["reinforce", "vpg", "ppo"], help="Algorithm to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Specific path to model .pth file. If None, picks latest in ./models")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the environment visually")
    parser.add_argument("--wandb", action="store_true", help="Log results to WandB")
    
    args = parser.parse_args()

    # Determine model path
    model_path = args.model
    if model_path is None:
        model_path = get_latest_model(args.algo)
        if model_path is None:
            print(f"No models found for {args.algo} in ./models directory.")
            exit(1)
    
    evaluate(
        algo=args.algo, 
        model_path=model_path, 
        episodes=args.episodes, 
        render=args.render,
        use_wandb=args.wandb
    )