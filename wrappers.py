import cv2
import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, ObservationWrapper
from collections import deque
from gymnasium import spaces

class PreprocessWrapper(ObservationWrapper):
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
        grayscale = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. Crop (remove status bar)
        cropped = grayscale[12:, :] 
        # 3. Resize
        resized = cv2.resize(cropped, self.resize, interpolation=cv2.INTER_AREA)
        # 4. Normalise and Expand dims to (1, H, W)
        normalised = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalised, axis=0)

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