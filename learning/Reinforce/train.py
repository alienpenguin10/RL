"""
imports the agent, creates the environment, and runs the training loop. It calls agent.select_action() and agent.update()
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from agents import REINFORCEAgent

# Create environment
env = gym.make("CarRacing-v3",render_mode=None,lap_complete_percent=0.95,domain_randomize=False,continuous=True)

# Get state size and action size
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]

# Create agent
# Initialize policy network parameters θ randomly
# Set learning rate α, discount factor γ
learning_rate = 0.0003
discount_factor = 0.99
agent = REINFORCEAgent(learning_rate, discount_factor)

# Initialize lists to track rewards
episode_rewards = []
episode_numbers = []

def plot_rewards(episode_rewards, episode_numbers, window=100, save_path="models/reward_plot.png"):
    """
    Plot episode rewards with moving average
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(episode_numbers, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Calculate and plot moving average
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        moving_avg_episodes = episode_numbers[window-1:]
        plt.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window} episodes)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

num_episodes = 2500
for episode in range(num_episodes):
    # Generate an episode following policy π_θ:
    # Initialize state s_0
    state, info = env.reset()
    terminal = False
    truncated = False
    episode_reward = 0
    episode_steps = 0
    while not (terminal or truncated):
        # Sample action a_t ~ π_θ(a_t | s_t)
        action = agent.select_action(state)
        # Execute action a_t, observe reward r_t and next state s_{t+1}
        next_state, reward, terminal, truncated, info = env.step(action)
        # Store (s_t, a_t, r_t)
        agent.store_transition(state, action, reward)
        # Update the current state and episode reward
        state = next_state
        episode_reward += reward
        episode_steps += 1
    
    # Perform learning update after episode
    policy_loss, value_loss = agent.update()
    
    # Track rewards for plotting
    episode_rewards.append(episode_reward)
    episode_numbers.append(episode)
    
    print(f"Episode {episode} Reward: {episode_reward} Steps: {episode_steps} Policy Loss: {policy_loss} Value Loss: {value_loss}")

    if episode % 10 == 0:
        agent.save_model(f"models/model_{episode}.pth")
        print(f"Episode {episode} Saved Model")
        # Update reward plot
        plot_rewards(episode_rewards, episode_numbers, save_path=f"models/reward_plot_episode_{episode}.png")
        print(f"Episode {episode} Updated Reward Plot")

# Generate final reward plot
print("\nGenerating final reward plot...")
plot_rewards(episode_rewards, episode_numbers, save_path="models/reward_plot_final.png")
print("Training complete! Final reward plot saved to models/reward_plot_final.png")

# # For each time step t in the episode:
#     #     Compute return G_t = Σ(k=t to T-1) γ^(k-t) * r_k
#     discounted_rewards = agent.compute_returns(rewards)

# For each time step t in the episode:
    #     Compute gradient: ∇_θ J(θ) ≈ G_t * ∇_θ log π_θ(a_t | s_t)
    #     Update parameters: θ ← θ + α * ∇_θ J(θ)
    #agent.update(states, actions, discounted_rewards)

#TODO: Frame Stacking: Stack 4 consecutive frames to capture motion​
#TODO: Grayscale: Convert to grayscale to reduce computation​
#TODO: Frame Skip: Only act every 4 frames for faster training
#TODO: Reward Shaping: Car Racing's default rewards are sparse - consider shaping them
