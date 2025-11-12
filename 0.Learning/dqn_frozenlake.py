import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import random
import torch
from torch import nn
import torch.nn.functional as F

#Define the model
class DQN(nn.Module):
    def __init__(self, input_states, hidden_nodes, out_actions):
        super().__init__()

        # Define network layers : 1- input layer, 2- hidden layer, 2- output layer
        self.fc1 = nn.Linear(input_states, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, out_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Define memory for experience replay
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# FrozenLake Deep Q-learning
class FrozenLakeDQL():
    def __init__(self):       
        # Initialize the environment. The 'FrozenLake-v1' environment is a grid-based game where the agent must navigate from start to goal.
        # 'is_slippery=True' makes the environment stochastic, adding randomness to the agent's actions.
        self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human') # Slippery -> doesn't always honor the step action
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        # Set the learning rate (alpha), discount factor (gamma), and number of episodes.
        self.alpha = 0.9
        self.gamma = 0.9
        self.network_sync_rate = 10 # Number steps taken before syncing the target network with the policy network
        self.episodes = 100
        self.mini_batch_size = 32
        self.epsilon = 1
        self.epsilon_decay_rate = 0.001 # Epsilon decay rate - 1/0.0001 = 10,000
        # Loop until the episode is finished.
        self.epsilon_history = []
        # Initialize a random number generator.
        self.rng = np.random.default_rng()

        
        self.replay_memory = ReplayMemory(1000)
        
        self.actions = ['left', 'down', 'right', 'up']
    
    def train(self):
        # Step1
        policy_net = DQN(self.num_states, self.num_states, self.num_actions)
        
        # Step2
        target_net = DQN(self.num_states, self.num_states, self.num_actions) 
        target_net.load_state_dict(policy_net.state_dict()) # Set the target network weights to be the same as the policy network weights. 

        print("Policy Net: ", policy_net)
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
        rewards_per_episode = np.zeros(self.episodes)

        steps_count = 0 # For step 10: syncing

        # Training loop over the specified number of episodes.
        for i in range(self.episodes):
            state = self.env.reset()[0]  # Reset the environment to the initial state.
            terminated = False # Becomes true when the episode ends (goal reached or fell in a hole).
            truncated = False # True when the episode is truncated (actions > 200).
            reward_for_current_episode = 0
            

            while (not terminated and not truncated):
                # Epsilon-greedy action selection: either explore (random action) or exploit (best known action).
                if self.rng.random() < epsilon: 
                    action = self.env.action_space.sample() # Random action: 0-left, 1-down, 2-right, 3-up
                else:
                    action = np.argmax(policy_net(self.encode_state(state, self.num_states)))
                
                # Take the action and observe the new state and reward.
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Save the experience to the replay memory
                self.replay_memory.push((state, action, new_state, reward, terminated))

                # Transition to the new state.
                state = new_state

                # Accumulate the reward for the episode.
                reward_for_current_episode += reward
                
                # Increment the step counter:
                steps_count += 1
            
            # Keep track of the rewards collected per episode
            rewards_per_episode[i] = reward_for_current_episode

            # Check if enough experiences are stored in the replay memory amd if at least 1 reward has been collected
            if len(self.replay_memory.memory) >= self.mini_batch_size and np.sum(rewards_per_episode > 0):
                mini_batch = self.replay_memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_net, target_net)

            # Decay epsilon
            epsilon = max(self.epsilon - 1/self.episodes, 0)
            self.epsilon_history.append(self.epsilon)

            # Copy policy network to target network after a certain number of steps
            if steps_count > self.network_sync_rate:
                target_net.load_state_dict(policy_net.state_dict())
                steps_count = 0
        # Close the environment after training.
        self.env.close()

    def optimize(self):
        pass

    def encode_state(self, state, num_states):
        pass
        return np.eye(num_states)[state]
    
    def test(self):
        pass