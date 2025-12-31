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
    # def __init__(self, env):
    #     super().__init__(env)
    #     assert isinstance(env.observation_space, Box), "Expected Box observation space"
    #     assert len(env.observation_space.shape) == 3, "Expected 3D observation space (H, W, C)"
    #     obs_shape = (84, 96)  # Grayscale frames of shape (H: 84, W: 96)
    #     self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def __init__(self, env, obs_shape=(84, 96)):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), "Expected Box observation space"
        self.obs_dims = (1, *obs_shape) # Grayscale frames of shape (1, H: 84, W: 96) pytorch format
        self.observation_space = Box(low=0.0, high=1.0, shape=self.obs_dims, dtype=np.float32) # Normalized to [0, 1]

    # def observation(self, obs):
    #     # Convert to grayscale
    #     obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    #     # Ensure observation is resized to default size 96x96
    #     obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_AREA)
    #     # Resize to 84x96 by cropping out driving info at bottom
    #     obs = obs[0:84, :]
    #     return obs

    def observation(self, obs):
        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Ensure observation is resized to default size 96x96
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_AREA)
        # Resize to 84x96 by cropping out driving info at bottom
        obs = obs[0:84, :]
        # Normalize to [0, 1] and add channel dimension
        obs = obs.astype(np.float32) / 255.0
        obs = np.expand_dims(obs, axis=0)  # Shape: (1, H, W)
        return obs

class FrameStack(gym.ObservationWrapper):
    # def __init__(self, env, num_frames, skip_frames = 0):
    #     super().__init__(env)
    #     self.num_frames = num_frames

    #     """
    #         skip_frames: Number of frames to skip between stacked frames
    #         eg. num_frames=4 with skip_frames=2 will stack frames [f0, f2, f4, f6]
    #     """
    #     self.skip_frames = skip_frames
    #     self.queue_len = num_frames * (skip_frames+1) - skip_frames if skip_frames > 0 else num_frames
    #     self.frames = deque([], maxlen=self.queue_len)

    #     assert isinstance(env.observation_space, Box), "Expected Box observation space"
    #     assert (len(env.observation_space.shape) == 2 or len(env.observation_space.shape) == 3,
    #             "Expected grayscale frame observation space (H, W) or RGB frame observation space (H, W, C)")
    #     # Input: (H, W, C) -> Output:(1, C, H, W)
    #     stacked_obs_shape = (num_frames, env.observation_space.shape[0], env.observation_space.shape[1])
    #     self.observation_space = Box(low=0, high=255, shape=stacked_obs_shape, dtype=np.uint8)
    
    def __init__(self, env, num_frames=4, skip_frames=0):
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
        assert len(env.observation_space.shape) == 3, "Expected grayscale frame observation space (F, H, W)"
        # Input: (C, H, W) -> Output:(F*C, H, W)
        stacked_obs_dim = (num_frames*env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])
        self.observation_space = Box(low=0.0, high=1.0, shape=stacked_obs_dim, dtype=np.float32)

    def observation(self, obs):
        self.frames.append(obs)
        return self.get_stacked_frames()

    def reset(self, **kwargs):
        """
            Reset the environment and stack the initial frame
        """
        obs, info = self.env.reset(**kwargs)
        # resets so that for the very first step, the history will look like [frame0, frame0, frame0, frame0]
        # Then transitions to [frame0, frame0, frame0, frame1] after one step.
        for _ in range(self.queue_len):
            self.frames.append(obs)
        return self.get_stacked_frames(), info
    
    def get_stacked_frames(self):
        assert len(self.frames) == self.queue_len, "Not enough frames stacked"
        # Convert deque to list before slicing (deque doesn't support extended slicing)
        frames_list = list(self.frames)
        stacked_frames = frames_list[::self.skip_frames+1] if self.skip_frames > 0 else frames_list
        assert len(stacked_frames) == self.num_frames, f"Expected {self.num_frames} frames, got {len(stacked_frames)}"
        
        return np.expand_dims(np.concatenate(stacked_frames, axis=0), axis=0)

class SpeedInfoWrapper(gym.Wrapper):
    """ Wrapper to include car speed information """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Get speed from car's linear velocity
        vel = self.env.unwrapped.car.hull.linearVelocity
        speed = np.linalg.norm(vel)
        info["speed"] = speed
        return obs, reward, terminated, truncated, info

class ActionRepeatWrapper(gym.Wrapper):
    """ Wrapper to repeat the same action for a number of frames """
    def __init__(self, env, skip_frames=0):
        super().__init__(env)
        # Repeat action during skipped frames (0 skips = no repeat/perform once)
        self.repeat = skip_frames

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.repeat+1):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Prevent reward from falling below -5
        # total_reward = max(total_reward, -5.0)
        
        return observation, total_reward, terminated, truncated, info

class ActionRemapWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # We tell the agent: "You have 3 actions, all between -1 and 1"
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def action(self, action):
        # Map agent's [-1, 1] outputs to the environment's required format
        
        # Steering: [-1, 1] -> [-1, 1] (No change)
        steer = action[0]
        
        # Gas: [-1, 1] -> [0, 1]
        # We use (x + 1) / 2 so that an output of 0.0 (network init) becomes 0.5 (half gas)
        gas = (action[1] + 1) / 2.0
        
        # Brake: [-1, 1] -> [0, 1]
        brake = (action[2] + 1) / 2.0
        
        return np.array([steer, gas, brake], dtype=np.float32)
    
class PolicyActionMapWrapper(gym.ActionWrapper):
    def __init__(self, env, policy_output_dim=3):
        super().__init__(env)
        # Car Racing expects 3 actions: (steering [-1,1], gas [0,1], brake [0,1])
        self.action_space = Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64)
        # Agent can output either 2 or 3 actions
        self.policy_output_dims = policy_output_dim

    def action(self, policy_output):
        # Steering: [-1, 1] -> [-1, 1] (No change)
        steer = policy_output[0]
        
        if self.policy_output_dims == 3:
            # Gas and Brake directly from policy
            gas = policy_output[1]
            brake = policy_output[2]
        else:
            # Speed: [-1, 1] -> Gas/Brake [0, 1]
            speed = policy_output[1]
            if speed > 0:
                gas = speed
                brake = 0.0
            else:
                gas = 0.0
                brake = -speed
        
        return np.array([steer, gas, brake], dtype=np.float32)