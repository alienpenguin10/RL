import os
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from env import env_creator

register_env("CarRacingFlat", env_creator)

config = (
    SACConfig()
    .environment("CarRacingFlat")
    .env_runners(num_env_runners=1)
    .training(
        gamma=0.9,
        actor_lr=0.001,
        critic_lr=0.002,
        train_batch_size_per_learner=32,
    )
)

algo = config.build_algo()

for i in range(101):
    result = algo.train()
    reward = result.get("env_runners/episode_return_mean", "N/A")
    print(f"Iteration {i}: reward={reward}")
    checkpoint_path = os.getcwd() + "/checkpoints"

    if i % 10 == 0 and i != 0:
        checkpoint = algo.save(f"{checkpoint_path}/{i}")
        print(f"Checkpoint saved at {checkpoint}")

algo.stop()
