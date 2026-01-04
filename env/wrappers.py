from collections import deque
from typing import Literal, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class PreprocessWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize: Tuple[int, int] = (84, 84),
        grayscale: bool = True,
        crop_top: int = 0,
        crop_bottom: int = 12,
        normalize: bool = True,
    ):
        super().__init__(env)
        self.resize = resize
        self.grayscale = grayscale
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.normalize = normalize

        # Determine channels: 1 for Grayscale, 3 for RGB
        self.channels = 1 if grayscale else 3

        # Define shape as (C, H, W) for PyTorch
        low, high = (0.0, 1.0) if normalize else (0, 255)
        dtype = np.float32 if normalize else np.uint8

        self.observation_space = Box(
            low=low,
            high=high,
            shape=(self.channels, resize[0], resize[1]),
            dtype=dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.grayscale:
            # Convert to gray -> Shape: (H, W)
            processed = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            # Keep RGB -> Shape: (H, W, 3)
            processed = obs

        h = processed.shape[0]
        crop_end = h - self.crop_bottom if self.crop_bottom > 0 else h
        cropped = processed[self.crop_top : crop_end, :]

        resized = cv2.resize(cropped, self.resize[::-1], interpolation=cv2.INTER_AREA)

        if self.normalize:
            result = resized.astype(np.float32) / 255.0
        else:
            result = resized

        # 5. Channel First Formatting (H, W, C) -> (C, H, W)
        if self.grayscale:
            # Add channel dimension: (H, W) -> (1, H, W)
            return np.expand_dims(result, axis=0)
        else:
            # Transpose dimensions: (H, W, 3) -> (3, H, W)
            return np.transpose(result, (2, 0, 1))


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_frames: int = 4, skip_frames: int = 0):
        super().__init__(env)
        self.num_frames = num_frames
        self.skip_frames = skip_frames

        # Calculate number of frames to skip between stacked frames
        # eg. num_frames=4 with skip_frames=2 will stack frames [f0, f2, f4, f6]
        self.buffer_len = (
            num_frames * (skip_frames + 1) - skip_frames
            if skip_frames > 0
            else num_frames
        )
        self.frames = deque(maxlen=self.buffer_len)

        channels, height, width = env.observation_space.shape
        self.observation_space = Box(
            low=env.observation_space.low.flat[0],
            high=env.observation_space.high.flat[0],
            shape=(num_frames * channels, height, width),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill buffer with initial frame
        for _ in range(self.buffer_len):
            self.frames.append(obs)
        return self._get_stacked_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_observation(), reward, terminated, truncated, info

    def _get_stacked_observation(self) -> np.ndarray:
        frames_list = list(self.frames)
        if self.skip_frames > 0:
            selected = frames_list[:: self.skip_frames + 1]
        else:
            selected = frames_list

        # Concatenate along channel dimension (axis 0)
        return np.concatenate(selected, axis=0)


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        repeat: int = 4,
        clip_reward: float | None = None,
    ):
        super().__init__(env)
        self.repeat = max(1, repeat)
        self.clip_reward = clip_reward

    def step(self, action):
        obs = None
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if self.clip_reward is not None:
            total_reward = max(total_reward, self.clip_reward)

        return obs, total_reward, terminated, truncated, info


class ActionMapWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        use_throttle: bool = True,
    ):
        super().__init__(env)
        self.use_throttle = use_throttle

        if use_throttle:
            # 2D: [steering, throttle] both in [-1, 1]
            self.action_space = Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # 3D: all in [-1, 1], will be remapped
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        if self.use_throttle:
            steering = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], -1.0, 1.0)

            if throttle >= 0:
                gas, brake = throttle, 0.0
            else:
                gas, brake = 0.0, -throttle

        else:
            # Map [-1, 1] to appropriate ranges
            steering = action[0]
            gas = (action[1] + 1) / 2.0  # [-1,1] -> [0,1]
            brake = (action[2] + 1) / 2.0  # [-1,1] -> [0,1]

        return np.array([steering, gas, brake], dtype=np.float32)

    def reverse_action(self, env_action: np.ndarray) -> np.ndarray:
        steering, gas, brake = env_action

        if self.use_throttle:
            throttle = gas if gas > brake else -brake
            return np.array([steering, throttle], dtype=np.float32)

        else:
            # Reverse: [0,1] -> [-1,1]
            return np.array(
                [
                    steering,
                    gas * 2.0 - 1.0,
                    brake * 2.0 - 1.0,
                ],
                dtype=np.float32,
            )


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


class SpeedInfoWrapper(gym.Wrapper):
    """Wrapper to include car speed information"""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Get speed from car's linear velocity
        vel = self.env.unwrapped.car.hull.linearVelocity
        speed = np.linalg.norm(vel)
        info["speed"] = speed
        return obs, reward, terminated, truncated, info
