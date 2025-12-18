import cv2
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, ObservationWrapper
from collections import deque

class PreprocessCarRacing(ObservationWrapper):
    def __init__(self, env, resize=(84, 84)):
        super().__init__(env)
        self.resize = resize
        # Shape is (1, H, W) -> PyTorch friendly
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1, resize[0], resize[1]), 
            dtype=np.float32
        )

    def observation(self, obs):
        # 1. Grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. Crop (remove status bar)
        cropped = gray[12:, :] 
        # 3. Resize
        resized = cv2.resize(cropped, self.resize, interpolation=cv2.INTER_AREA)
        # 4. Normalize and Expand dims to (1, H, W)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

class FrameStack(Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        c, h, w = env.observation_space.shape
        # Stack channels: (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(c * num_stack, h, w),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info

    def _get_obs(self):
        # Concatenate along channel dimension (axis 0)
        return np.concatenate(self.frames, axis=0)

class RepeatAction(gym.Wrapper):
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
        
        # Soft clip negative rewards to prevent agent from "giving up"
        # CarRacing gives -0.1 per frame. With skip=4, that's -0.4.
        # If the car spins, it might get -100. We cap the lower bound slightly.
        total_reward = max(total_reward, -5.0)
        
        return obs, total_reward, term, trunc, info