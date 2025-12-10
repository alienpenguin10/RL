import gymnasium as gym
from gymnasium.spaces import Box
import cv2
import numpy as np
from collections import deque

class ProcessedFrame(gym.ObservationWrapper):
    """
        Process the raw RGB frames to grayscale and resize to 84x96
        Grayscale to reduce channel dimension of each frame
        Resize to (H: 84, W: 96) to remove driving info at bottom
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Expected Box observation space"
        assert len(env.observation_space.shape) == 3, "Expected 3D observation space (H, W, C)"

        obs_shape = (84, 96)  # Grayscale frames of shape (H: 84, W: 96)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs):
        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Ensure observation is resized to default size 96x96
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_AREA)
        # Resize to 84x96 by cropping out driving info at bottom
        obs = obs[0:84, :]
        return obs

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_frames, skip_frames = 0):
        super().__init__(env)
        self.num_frames = num_frames

        """
            skip_frames: Number of frames to skip between stacked frames
            eg. num_frames=4 with skip_frames=2 will stack frames [f0, f2, f4, f6]
        """
        self.skip_frames = skip_frames
        self.queue_len = num_frames * (skip_frames+1) - skip_frames if skip_frames > 0 else num_frames
        self.frames = deque([], maxlen=self.queue_len)

        assert isinstance(env.observation_space, Box), "Expected Box observation space"
        assert (len(env.observation_space.shape) == 2 or len(env.observation_space.shape) == 3,
                "Expected grayscale frame observation space (H, W) or RGB frame observation space (H, W, C)")

        stacked_obs_shape = (num_frames, env.observation_space.shape[0], env.observation_space.shape[1])
        self.observation_space = Box(low=0, high=255, shape=stacked_obs_shape, dtype=np.uint8)

    def observation(self, obs):
        self.frames.append(obs)
        return self.get_stacked_frames()

    def reset(self, **kwargs):
        """
            Reset the environment and stack the initial frame
        """
        obs = self.env.reset(**kwargs)[0]
        for _ in range(self.queue_len):
            self.frames.append(obs)
        return self.get_stacked_frames()
    
    def get_stacked_frames(self):
        assert len(self.frames) == self.queue_len, "Not enough frames stacked"
        # Convert deque to list before slicing (deque doesn't support extended slicing)
        frames_list = list(self.frames)
        stacked_frames = frames_list[::self.skip_frames+1] if self.skip_frames > 0 else frames_list
        assert len(stacked_frames) == self.num_frames, f"Expected {self.num_frames} frames, got {len(stacked_frames)}"
        
        return np.stack(stacked_frames, axis=0)