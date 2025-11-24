import os

from env import FlattenObservation, env_creator
import gymnasium as gym
import torch
from ray.rllib.algorithms.sac import SACConfig, SAC
from ray.tune.registry import register_env
import time


register_env("CarRacingFlat", env_creator)


def visualize_episode(checkpoint_path, num_episodes=1, render=True):
    config = (
        SACConfig()
        .environment("CarRacingFlat", env_config={"render_mode": "human" if render else None})
        .env_runners(num_env_runners=1)
        .evaluation(
            evaluation_interval=None,
            evaluation_num_env_runners=0,
        )
        .framework("torch")
    )

    algo = config.build_algo()
    algo.restore(checkpoint_path)

    env = gym.make("CarRacing-v3", render_mode="human" if render else None)
    env = FlattenObservation(env)
    module = algo.get_module()

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not (done or truncated):
            obs_batch = torch.tensor([obs], dtype=torch.float32)

            with torch.no_grad():
                module_output = module.forward_inference({"obs": obs_batch})
                if isinstance(module_output, dict):
                    if "actions" in module_output:
                        action = module_output["actions"][0].numpy()
                    else:
                        action = module_output.get("action_dist_inputs", module_output)[0].numpy()
                else:
                    action = module_output[0].numpy()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if render:
                time.sleep(0.01)

            if step_count % 50 == 0:
                print(f"Step {step_count}, Current reward: {total_reward:.2f}")

        print(f"Episode finished! Total reward: {total_reward:.2f}, Steps: {step_count}")

    env.close()
    algo.stop()


if __name__ == "__main__":
    checkpoint_path = os.getcwd() + "/checkpoints/100"
    visualize_episode(checkpoint_path, num_episodes=3, render=True)
