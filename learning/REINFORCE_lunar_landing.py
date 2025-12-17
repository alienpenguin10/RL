# Import pytorch, numpy, openaigym
import numpy as np
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import sys
print("All good")
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

# Define hyperparameters
gamma = 0.99

# State size - 8 -> Position, Velocity, Angle, Angular Velocity, Leg 1, Leg 2, Leg 3, Leg 4
# Action size - 4 -> 0: do nothing, 1: fire left orientation engine, 2: fire main engine, 3: fire right orientation engine
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define neural network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    # Define policy forward
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=0)

policy_network = PolicyNetwork()



# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy_network.parameters(), lr=0.01)

# Update policy
def update_policy(states, actions, discounted_rewards):
    # log_probs = torch.stack(log_probs)
    # loss = -torch.mean(log_probs * (sum(rewards)))
    action_probs = policy_network(torch.tensor(states, dtype=torch.float32))
    # Calculate the loss
    loss = torch.log(action_probs.gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1).float())
    loss = loss * torch.tensor(discounted_rewards, dtype=torch.float32)
    loss = -torch.sum(loss)
    # Calculate the gradients and update the policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for episode in range(100):
    # Initialise empty lists to store state, action, and reward
    states = []
    actions = []
    rewards = []
    log_probs = []
    
    # Choose initial state
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0

    # Run the episode
    while True:
        # Get probabilities from the policy network
        action_probs = policy_network.forward(torch.tensor(state, dtype=torch.float32))

        # Choose action based on the probabilities
        choosen_action = np.random.choice(action_size, p=action_probs.detach().numpy())
        log_prob = torch.log(action_probs[choosen_action])

        # Perform action and observe the reward
        next_state, reward, terminated, truncated, info = env.step(choosen_action)

        # Store the current state, action, and reward
        states.append(observation)
        actions.append(choosen_action)
        rewards.append(reward) 
        log_probs.append(log_prob)

        # Update the current state and episode reward
        state = next_state
        episode_reward += reward
        episode_steps += 1

        if terminated or truncated:
            print(f"Episode {episode} finished after {episode_steps} steps with a reward of {episode_reward}")
            break
        
    # Calculate return of a single episode
    # discounted_reward = 0
    # for i in range(len(rewards)):
    #     discounted_reward += rewards[i] * gamma ** i
        # Calculate the discounted rewards for each step in the episode
    discounted_rewards = np.zeros_like(rewards)
    running_total = 0
    for i in reversed(range(len(rewards))):
        running_total = running_total * gamma + rewards[i]
        discounted_rewards[i] = running_total
    
    # Normalize the discounted rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    
    # Update the policy
    update_policy(states, actions, discounted_rewards)

    #Save the PyTorch model
    torch.save(policy_network.state_dict(), "policy_network.pth")

env.close()