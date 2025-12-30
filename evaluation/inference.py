import gymnasium as gym
import numpy as np
from agents.reinforce import REINFORCEAgent
from agents.vpg import VPGAgent
from agents.ppo_old import PPOAgent
import torch
import os
import time

def inference(env_name="CarRacing-v3", algo="reinforce", model_path=None):
    """
    Visualize a trained agent running in the environment using deterministic inference.
    The agent uses its learned policy to select actions deterministically.
    
    Args:
        env_name: Name of the gymnasium environment
        algo: Algorithm type ("reinforce", "vpg", or "ppo")
        model_path: Path to the trained model file
    """
    if model_path is None:
        raise ValueError("model_path must be provided")
    
    # Initialize agent
    if algo == "reinforce":
        agent = REINFORCEAgent(learning_rate=0.001)
    elif algo == "vpg":
        agent = VPGAgent(learning_rate=0.0003)
    elif algo == "ppo":
        agent = PPOAgent(learning_rate=0.0003, clip_ratio=0.2)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        agent.policy_network.eval()
        print(f"Loaded {algo} model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Visualizing {algo} agent on {env_name} using device: {agent.device}\n")
    
    # Create environment with rendering enabled
    try:
        env = gym.make(env_name, continuous=True, render_mode="human")
    except gym.error.Error as e:
        print(f"Environment {env_name} not found or error creating it: {e}")
        print("Make sure gymnasium[box2d] is installed.")
        return
    
    # Run one episode with deterministic inference
    state, _ = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done:
        # Preprocess state and get action from policy network
        state_tensor = agent.preprocess_state(state)
        
        with torch.no_grad():
            action = agent.policy_network.infer(state_tensor).squeeze(0)
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        state = next_state
        episode_reward += reward
        steps += 1
        
        # Small delay for visualization
        time.sleep(0.01)
        
        # Safety break for very long episodes
        if steps > 2000:
            print(f"Episode truncated due to length.")
            done = True
    
    print(f"\nEpisode finished! Reward: {episode_reward:.2f} | Steps: {steps}")
    
    env.close()
    print("Visualization complete!")

def get_available_models(model_dir="./models"):
    """Get all available model files in the models directory."""
    if not os.path.exists(model_dir):
        return []
    return [f for f in os.listdir(model_dir) if f.endswith(".pth")]

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    print("\n=== RL Agent Inference ===")
    
    # Step 1: Choose algorithm
    print("\nAvailable algorithms:")
    print("1. REINFORCE")
    print("2. VPG (Vanilla Policy Gradient)")
    print("3. PPO (Proximal Policy Optimization)")
    
    while True:
        choice = input("\nSelect algorithm (1-3): ").strip()
        if choice == "1":
            algo = "reinforce"
            break
        elif choice == "2":
            algo = "vpg"
            break
        elif choice == "3":
            algo = "ppo"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Step 2: Choose model file
    models = get_available_models()
    
    if not models:
        print(f"\nNo model files found in ./models directory.")
        print("Please train a model first using train.py")
        exit(1)
    
    # Filter models by algorithm
    algo_models = [m for m in models if m.startswith(algo)]
    
    if not algo_models:
        print(f"\nNo {algo.upper()} models found in ./models directory.")
        print(f"Available models: {', '.join(models)}")
        exit(1)
    
    print(f"\nAvailable {algo.upper()} models:")
    for i, model in enumerate(algo_models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(algo_models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(algo_models):
                model_path = os.path.join("./models", algo_models[idx])
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(algo_models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Run inference
    print(f"\n--- Visualizing {algo.upper()} agent ---")
    inference(algo=algo, model_path=model_path)

