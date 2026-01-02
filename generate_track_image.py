import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # Add this import

env = gym.make("CarRacing-v3", render_mode="rgb_array")
env.reset()

# Step the environment forward a few times to move the car onto the track
for _ in range(20):  # You can adjust the number of steps
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))  # accelerate forward

img = env.render()

plt.imsave("car_racing_track.png", img)
env.close()