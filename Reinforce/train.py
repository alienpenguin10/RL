"""
imports the agent, creates the environment, and runs the training loop. It calls agent.select_action() and agent.update()
"""
import gymnasium as gym
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


num_episodes = 10
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
    print(f"Episode {episode} Reward: {episode_reward} Steps: {episode_steps} Policy Loss: {policy_loss} Value Loss: {value_loss}")


    if episode % 10 == 0:
        agent.save_model(f"models/model_{episode}.pth")
        print(f"Episode {episode} Saved Model")


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
