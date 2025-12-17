import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

"""
States: (P, V, A, AV)
Actions : 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
Observation space: 64
There will be infinitely possible values -> chopped it up in ranges
"""

# Initialize the environment. The 'FrozenLake-v1' environment is a grid-based game where the agent must navigate from start to goal.
# 'is_slippery=True' makes the environment stochastic, adding randomness to the agent's actions.
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human') # Slippery -> doesn't always honor the step action

# Initialize the Q-table to zeros. The shape is (number of states, number of actions).
q_table = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array

# Set the learning rate (alpha), discount factor (gamma), and number of episodes.
alpha = 0.9
gamma = 0.9
episodes = 100

# Array to store the total rewards per episode.
rewards_per_episode = np.zeros(episodes)

# Epsilon is the probability of exploring (taking a random action). It decays over time to favor exploitation (using learned actions).
epsilon = 1
epsilon_decay_rate = 0.001 # Epsilon decay rate - 1/0.0001 = 10,000

# Initialize a random number generator.
rng = np.random.default_rng()

# Training loop over the specified number of episodes.
for i in range(episodes):
    state = env.reset()[0]  # Reset the environment to the initial state.
    terminated = False # Becomes true when the episode ends (goal reached or fell in a hole).
    truncated = False # True when the episode is truncated (actions > 200).
    reward_for_current_episode = 0
    # Loop until the episode is finished.
    while (not terminated and not truncated):
        # Epsilon-greedy action selection: either explore (random action) or exploit (best known action).
        if rng.random() < epsilon: 
            action = env.action_space.sample() # Random action: 0-left, 1-down, 2-right, 3-up
        else:
            action = np.argmax(q_table[state, :])
        
        # Take the action and observe the new state and reward.
        new_state, reward, terminated, truncated, _ = env.step(action)
        
        # Update the Q-value using the Q-learning formula.
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        # Transition to the new state.
        state = new_state

        # Accumulate the reward for the episode.
        reward_for_current_episode += reward

    rewards_per_episode[i] = reward_for_current_episode
    # Decay epsilon to reduce the probability of random actions as learning progresses.
    epsilon = max(epsilon - epsilon_decay_rate, 0)

# Close the environment after training.
env.close()

# Calculate the cumulative sum of rewards for plotting.
sum_rewards = np.zeros(episodes)
for t in range(episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[0:t+1])

# Plot the cumulative rewards over episodes.
plt.plot(sum_rewards)
plt.savefig('frozen_lake8x8.png')

# Save the Q-table to a file for later use.
with open('frozen_lake_q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)
print("Q Table saved")
