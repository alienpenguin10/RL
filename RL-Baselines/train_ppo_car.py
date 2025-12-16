import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env


# Wrapper to normalize image observations from uint8 [0,255] to float32 [0,1]
class NormalizeImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update observation space to reflect float32 type
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype='float32'
        )
    
    def observation(self, obs):
        # Convert uint8 [0,255] to float32 [0,1]
        return obs.astype('float32') / 255.0


# Create wrapped environment factory function
def make_car_racing_env(config):
    env = gym.make("CarRacing-v3")
    env = NormalizeImageWrapper(env)
    return env


# Register the custom environment
register_env("CarRacing-Normalized", make_car_racing_env)

config = PPOConfig()
config.environment("CarRacing-Normalized")
config.env_runners(num_env_runners=2)

# Update training parameters for CarRacing (more complex than CartPole)
config.training(
    gamma=0.99,  # Higher discount for longer episodes
    lr=0.0003,   # Lower learning rate for stability
    kl_coeff=0.2,
    train_batch_size_per_learner=4000,  # Larger batch for image-based learning
    minibatch_size=128,  # Fixed: use minibatch_size instead of sgd_minibatch_size
    num_sgd_iter=10,
)

# Configure resources
config.resources(num_gpus=0)

# Build the algorithm
algo = config.build()

# Train for multiple iterations
for i in range(100):  # Run 100 training iterations
    result = algo.train()
    episode_reward = result.get('env_runners/episode_return_mean', 'N/A')
    print(f"Iteration {i+1}: episode_reward_mean = {episode_reward}")
    
    # Optionally save checkpoints
    if (i + 1) % 10 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved at: {checkpoint_dir}")