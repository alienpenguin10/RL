import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

"""
States: (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)
Actions : 0: Push left, 1: Push right 
Observation space: 64
There will be infinitely possible values -> chopped it up in ranges
N.B. There is no min and max range for Cart Velocity and Pole Angular Velocity
There will be infinitely possible values -> chopped it up in ranges
"""

env = gym.make('CartPole-v1', render_mode='human') # CartPole-v1 is a 2D continuous space where the agent must balance a pole on a cart.
pos_space = np.linspace(env.observation_space.low[0]/2, env.observation_space.high[0]/2, 10) # Between -4.8 and 4.8
velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 10) # Between -3.4 and 3.4
angle_space = np.linspace(env.observation_space.low[2]/2, env.observation_space.high[2]/2, 10) # Between -0.418 and 0.418
angular_space = np.linspace(env.observation_space.low[3], env.observation_space.high[3], 10) # Between -3.4 and 3.4

# linspace(1,3,3) -> [1,2,3]
# +1 to the length of the space to account for the case where the value is greater than the maximum value in the space
q_table = np.zeros((len(pos_space)+1, len(velocity_space)+1, len(angle_space)+1, len(angular_space)+1, env.action_space.n)) # init a 10x10x10x10x2 array

# Set the learning rate (alpha), discount factor (gamma), and number of episodes.
alpha = 0.9
gamma = 0.9
episodes = 100

# Array to store the total rewards per episode.
rewards_per_episode = np.zeros(episodes)

epsilon = 1
epsilon_decay_rate = 0.001 # Epsilon decay rate - 1/0.0001 = 10,000

# Initialize a random number generator.
rng = np.random.default_rng()

# Training loop over the specified number of episodes.
for i in range(episodes):
    state = env.reset()[0]  # Reset the environment to the initial state.
    pos_state = np.digitize(state[0], pos_space) # find the index of the position in the pos_space array
    velocity_state = np.digitize(state[1], velocity_space) # find the index of the velocity in the val_space array
    angle_state = np.digitize(state[2], angle_space) # find the index of the angle in the angle_space array
    angular_state = np.digitize(state[3], angular_space) # find the index of the angular velocity in the angular_velocity_space array
    terminated = False # Becomes true when the episode ends (goal reached or fell in a hole).
    truncated = False # True when the episode is truncated (actions > 200).
    reward_for_current_episode = 0
    while (not terminated and not truncated):
        # Epsilon-greedy action selection: either explore (random action) or exploit (best known action).
        if rng.random() < epsilon: 
            action = env.action_space.sample() # Random action: 0-Accelerate to the left, 1-Don't accelerate, 2-Accelerate to the right
        else:
            action = np.argmax(q_table[pos_state, velocity_state, angle_state, angular_state:])
        
        # Take the action and observe the new state and reward.
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_pos_state = np.digitize(new_state[0], pos_space)
        new_velocity_state = np.digitize(new_state[1], velocity_space)
        new_angle_state = np.digitize(new_state[2], angle_space)
        new_angular_state = np.digitize(new_state[3], angular_space)

        # Update the Q-value using the Q-learning formula.
        q_table[pos_state, velocity_state, angle_state, angular_state, action] = q_table[pos_state, velocity_state, angle_state, angular_state, action] + alpha * (reward + gamma * np.max(q_table[new_pos_state, new_velocity_state, new_angle_state, new_angular_state, :]) - q_table[pos_state, velocity_state, angle_state, angular_state, action])
        
        # Transition to the new state.
        state = new_state
        pos_state = new_pos_state
        velocity_state = new_velocity_state
        angle_state = new_angle_state
        angular_state = new_angular_state
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
plt.savefig('cartpole_q.png')

# Save the Q-table to a file for later use.
with open('cartpole_q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)
print("Q Table saved")