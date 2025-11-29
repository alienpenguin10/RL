"""
Car Racing Environment Simulation
Simulates the CarRacing-v3 environment from Gymnasium with random actions.
"""

import gymnasium as gym
import numpy as np


def run_car_racing(episodes=5, max_steps=1000, render_mode="human", continuous=True):
    """
    Run the Car Racing environment with random actions.
    
    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render_mode: Render mode ("human", "rgb_array", or None)
        continuous: If True, use continuous action space; if False, use discrete
    """
    # Create the environment
    env = gym.make(
        "CarRacing-v3",
        render_mode=render_mode,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=continuous
    )
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action mode: {'Continuous' if continuous else 'Discrete'}")
    print("\n" + "="*60)
    
    for episode in range(episodes):
        # Reset the environment
        observation, info = env.reset()
        total_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("-" * 60)
        
        done = False
        truncated = False
        
        while not (done or truncated) and step < max_steps:
            # Take a random action
            action = env.action_space.sample()
            
            # Execute the action
            observation, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                if continuous:
                    print(f"Step {step}: Reward={reward:.2f}, Total Reward={total_reward:.2f}, "
                          f"Action=[steering:{action[0]:.2f}, gas:{action[1]:.2f}, brake:{action[2]:.2f}]")
                else:
                    action_names = ["do nothing", "steer right", "steer left", "gas", "brake"]
                    print(f"Step {step}: Reward={reward:.2f}, Total Reward={total_reward:.2f}, "
                          f"Action={action_names[action]}")
        
        # Episode summary
        print(f"\nEpisode {episode + 1} finished after {step} steps")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Terminated: {done}, Truncated: {truncated}")
        print("="*60)
    
    env.close()
    print("\nSimulation completed!")


def main():
    """Main function to run the simulation."""
    print("Car Racing Environment - Random Actions")
    print("="*60)
    
    # Run with continuous action space (default)
    # Change render_mode to None or "rgb_array" if you don't want visualization
    run_car_racing(
        episodes=3,
        max_steps=1000,
        render_mode="human",  # Set to None if you don't want rendering
        continuous=True
    )


if __name__ == "__main__":
    main()

