import cv2
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, ObservationWrapper
from collections import deque
from gymnasium import spaces

class PreprocessWrapper(ObservationWrapper):
    def __init__(self, env, resize=(84, 84), grayscale=True):
        super().__init__(env)
        self.resize = resize
        self.grayscale = grayscale
        
        # Determine channels: 1 for Grayscale, 3 for RGB
        self.channels = 1 if grayscale else 3
        
        # Define shape as (C, H, W) for PyTorch
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.channels, resize[0], resize[1]), 
            dtype=np.float32
        )

    def observation(self, obs):
        # 1. Grayscale vs RGB
        if self.grayscale:
            # Convert to gray -> Shape: (H, W)
            processed = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            # Keep RGB -> Shape: (H, W, 3)
            processed = obs

        # 2. Crop (remove status bar)
        # Note: 'processed[12:, :]' works for both (H, W) and (H, W, C)
        cropped = processed[12:, :] 

        # 3. Resize
        # cv2.resize handles both 2D and 3D images correctly
        resized = cv2.resize(cropped, self.resize, interpolation=cv2.INTER_AREA)

        # 4. Normalise
        normalised = resized.astype(np.float32) / 255.0

        # 5. Channel First Formatting (H, W, C) -> (C, H, W)
        if self.grayscale:
            # Add channel dimension: (H, W) -> (1, H, W)
            return np.expand_dims(normalised, axis=0)
        else:
            # Transpose dimensions: (H, W, 3) -> (3, H, W)
            return np.transpose(normalised, (2, 0, 1))

class FrameStack(Wrapper):
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
        channels, height, width = env.observation_space.shape
        # Stack channels: (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(channels * num_frames, height, width),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self.observation(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(), reward, term, trunc, info

    def observation(self):
        # Concatenate along channel dimension (axis 0)
        return np.concatenate(self.frames, axis=0)

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            done = term or trunc
            if done:
                break
        
        # Prevent reward from falling below -5
        total_reward = max(total_reward, -5.0)
        
        return obs, total_reward, term, trunc, info


class ThrottleActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

    def action(self, action):
        steering = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], -1.0, 1.0)

        if throttle >= 0:
            gas = throttle
            brake = 0.0
        else:
            gas = 0.0
            brake = -throttle

        return np.array([steering, gas, brake], dtype=np.float32)

    def reverse_action(self, action):
        steering = action[0]
        gas = action[1]
        brake = action[2]

        if gas > brake:
            throttle = gas
        elif brake > gas:
            throttle = -brake
        else:
            throttle = 0.0

        return np.array([steering, throttle], dtype=np.float32)

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=0.05, skip_zoom_steps=50, patience=100):
        """
        Args:
            env: The environment to wrap.
            reward_scale: Factor to shrink rewards (stabilizes SAC critics).
            skip_zoom_steps: Number of steps to drop at start of episode (removes uncontrollable zoom).
            patience: Max steps allowed without visiting a new track tile before truncation.
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.skip_zoom_steps = skip_zoom_steps
        self.patience = patience
        
        # Internal state
        self.steps_without_progress = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Skip the first N frames where the camera zooms in 
        # and the car is uncontrollable.
        if self.skip_zoom_steps > 0:
            for _ in range(self.skip_zoom_steps):
                action_dim = self.env.action_space.shape[0]
                no_op = np.zeros(action_dim)
                
                obs, _, terminated, truncated, _ = self.env.step(no_op)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        
        self.steps_without_progress = 0
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Reward Scaling
        # Default rewards are ~7.0 for tiles. This brings them to ~0.35, 
        # which keeps Q-values in a range that neural nets like (-5 to 5).
        scaled_reward = reward * self.reward_scale

        # 2. Early Stopping (Patience)
        # CarRacing gives +Reward for tiles, -0.1 for time.
        # If reward > 0, we hit a new tile. Reset patience.
        # If reward <= 0, we are just burning time.
        if reward > 0:
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
            
        # If we haven't hit a new tile in 'patience' steps, kill the episode.
        # This prevents the agent from getting stuck in the grass for 500 steps
        # filling the buffer with garbage data.
        if self.steps_without_progress >= self.patience:
            truncated = True
            # Optional: Slight penalty for being too slow/stuck
            scaled_reward -= 1.0 

        return next_obs, scaled_reward, terminated, truncated, info