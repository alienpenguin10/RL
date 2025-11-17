import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(env.observation_space.shape),),
            dtype=np.float32
        )

    def observation(self, obs):
        obs_float = obs.astype(np.float32) / 255.0
        return obs_float.flatten()


def env_creator(env_config):
    env = gym.make("CarRacing-v3", render_mode=env_config.get("render_mode", None))
    return FlattenObservation(env)
