import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

"""
Actions : (Accelerate to the left, Don't accelerate, Accelerate to the right)
Observation space : (Car Position, Car Velocity) -> Divide position and velocity into segments 
There will be infinitely possible values -> chopped it up in ranges
N.B. There is min and max range for position and velocity
Rewards : -1 for each time step, 0 for reaching the goal
Episode ends when the car reaches the goal (poisition > 0.5) or after 200 time steps
"""

# Initialize the environment. The 'MountainCar-v0' environment is a 2D continuous space where the agent must drive a car up a hill.
env = gym.make('MountainCar-v0', render_mode='human') 
# Divide position and velocity into segments
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20) # Between -1.2 and 0.6
val_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20) # Between -0.07 and 0.07

# Initialize the Q-table to zeros. The shape is (number of states, number of actions).
q_table = np.zeros((len(pos_space), len(val_space), env.action_space.n)) # init a 20x20x3 array

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
    p_state = np.digitize(state[0], pos_space) # find the index of the position in the pos_space array
    v_state = np.digitize(state[1], val_space) # find the index of the velocity in the val_space array
    
    terminated = False # Becomes true when the episode ends (goal reached or fell in a hole).
    truncated = False # True when the episode is truncated (actions > 200).
    reward_for_current_episode = 0
    while (not terminated and not truncated):
        # Epsilon-greedy action selection: either explore (random action) or exploit (best known action).
        if rng.random() < epsilon: 
            action = env.action_space.sample() # Random action: 0-Accelerate to the left, 1-Don't accelerate, 2-Accelerate to the right
        else:
            action = np.argmax(q_table[p_state, v_state, :])
        
        # Take the action and observe the new state and reward.
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_p_state = np.digitize(new_state[0], pos_space)
        new_v_state = np.digitize(new_state[1], val_space)
        # Update the Q-value using the Q-learning formula.
        q_table[p_state,v_state, action] = q_table[p_state, v_state, action] + alpha * (reward + gamma * np.max(q_table[new_p_state, new_v_state, :]) - q_table[p_state, v_state, action])

        # Transition to the new state.
        state = new_state
        p_state = new_p_state
        v_state = new_v_state
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
plt.savefig('mountain_car_q.png')

# Save the Q-table to a file for later use.
with open('mountain_car_q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)
print("Q Table saved")