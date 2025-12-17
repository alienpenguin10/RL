import os
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import wandb

# Constants
VISUALIZE = False  # True: Single env with render, False: Parallel training (fast)
EVALUATE_ONLY = False # True: Run trained model (no training), False: Train model
PUSH_TO_WANDB = True  # True: Log to WandB, False: No logging
TOTAL_TIMESTEPS = 20_000_000
N_ENVS = 8  # Number of parallel environments when VISUALIZE=False

class SuperRacingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Define the new "Dict" observation space
        self.observation_space = spaces.Dict({
            # A tiny 64x64 binary image is enough for the road shape
            "image": spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.uint8),
            # Speed, Steering Angle, Angular Velocity, ABS sensors
            "sensors": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Process the observation
        processed_obs = self._process_obs(obs)
        
        # --- VISUALIZATION (If enabled) ---
        if VISUALIZE and self.env.render_mode == 'human':
            self._visualize_wrapper(processed_obs)
            
        return processed_obs, reward, terminated, truncated, info

    def _visualize_wrapper(self, obs):
        # 1. Prepare Mask Image
        # Convert (64, 64, 1) -> (256, 256, 3) for display
        mask = obs['image']
        # Remove channel dim for resize
        mask_2d = mask[:, :, 0]
        mask_large = cv2.resize(mask_2d, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        display_img = cv2.cvtColor(mask_large, cv2.COLOR_GRAY2BGR)
        
        # 2. Add Sensor Text
        sensors = obs['sensors']
        speed = sensors[0]
        angle = sensors[1]
        gyro = sensors[2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, f"Speed: {speed:.2f}", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Angle: {angle:.2f}", (10, 60), font, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Gyro:  {gyro:.2f}", (10, 90), font, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Agent 'Brain' View", display_img)
        cv2.waitKey(1)

    def _process_obs(self, obs):
        # 1. PROCESS IMAGE: Convert to Binary Road Mask
        mask = cv2.inRange(obs, np.array([100, 100, 100]), np.array([140, 140, 140]))
        
        # Resize to small 64x64 to save computation
        resized_mask = cv2.resize(mask, (64, 64))
        resized_mask = resized_mask[:, :, np.newaxis] # Add channel dim
        
        # 2. PROCESS SENSORS: Extract physics info from the environment internals
        car = self.env.unwrapped.car
        
        # Calculate speed magnitude from velocity vector
        speed = np.linalg.norm(car.hull.linearVelocity)
        
        # Get wheel angle (steering)
        angle = car.wheels[0].joint.angle
        
        # Angular velocity (are we spinning?)
        gyro = car.hull.angularVelocity
        
        # Create the sensor vector
        sensor_data = np.array([speed, angle, gyro, 0.0], dtype=np.float32)

        return {
            "image": resized_mask,
            "sensors": sensor_data
        }

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        # SB3 maintains a buffer of recent episode infos in self.model.ep_info_buffer
        # This is populated by the Monitor wrapper
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            # Extract rewards from the buffer
            rewards = [info["r"] for info in self.model.ep_info_buffer]
            if rewards:
                 mean_reward = np.mean(rewards)
                 # Log specifically as "reward_mean" to match other agents
                 wandb.log({"reward_mean": mean_reward, "global_step": self.num_timesteps})


from stable_baselines3.common.monitor import Monitor

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        # Wrap with Monitor BEFORE SuperRacingWrapper.
        # Monitor will record the 'true' reward (before shaping).
        # SuperRacingWrapper will add the shaping for the agent.
        # Result: WandB logs = True Reward, Agent sees = Shaped Reward.
        env = Monitor(env) 
        env = SuperRacingWrapper(env)
        return env
    return _init

import datetime
import glob

def evaluate_model(model_path=None):
    if model_path is None:
        # Find latest model
        list_of_files = glob.glob('ppo_binary_baseline_*.zip')
        if not list_of_files:
            print("No model files found (ppo_binary_baseline_*.zip). Cannot evaluate.")
            return
        latest_file = max(list_of_files, key=os.path.getctime)
        model_path = latest_file
    
    print(f"Loading model from {model_path} for evaluation...")
    env = make_env(render_mode="human")()
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    model = PPO.load(model_path, env=env)
    
    print("Starting evaluation loop... Press Ctrl+C to stop.")
    obs, _ = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        env.close()
        cv2.destroyAllWindows()

def main():
    if EVALUATE_ONLY:
        evaluate_model()
        return

    # WandB setup
    run_id = wandb.util.generate_id()
    if PUSH_TO_WANDB:
        wandb.init(
            project="rl-training",
            name="binary-ppo-baseline",
            id=run_id,
            sync_tensorboard=True,  # Auto-upload sb3 logs
            monitor_gym=True,       # Auto-upload gym videos
            save_code=True,
            config={
                "algorithm": "PPO",
                "policy": "MultiInputPolicy",
                "env": "CarRacing-v3",
                "observation": "Image(64x64) + Sensors",
                "total_timesteps": TOTAL_TIMESTEPS,
                "net_arch": "pi=[256, 256], vf=[256, 256]",
            }
        )

    print(f"Training Config: VISUALIZE={VISUALIZE}, PUSH_TO_WANDB={PUSH_TO_WANDB}")

    if VISUALIZE:
        # Single environment with rendering for debugging/visualization
        print("Running in visualization mode (single environment, render_mode='human')...")
        env = make_env(render_mode="human")()
        env = DummyVecEnv([lambda: env]) # Loop input for SB3
    else:
        # Parallel environments for speed
        print(f"Running in parallel mode ({N_ENVS} environments)...")
        # SubprocVecEnv runs each env in a separate process
        env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
        env = VecMonitor(env) # Helper to log episode rewards/lengths to tensorboard

    # Use a temporary directory for TensorBoard logs to avoid project clutter
    import tempfile
    import shutil
    
    # We will let the OS handle cleanup or we can try to clean it in finally block
    # For now, just putting it in /tmp/ (or OS equivalent) keeps the repo clean.
    tensorboard_log_dir = tempfile.mkdtemp(prefix="ppo_binary_baseline_logs_")
    print(f"TensorBoard logs will be stored in temporary dir: {tensorboard_log_dir}")

    # Use MultiInputPolicy
    # Note: tensorboard_log argument is required for sync_tensorboard=True to work
    model = PPO(
        "MultiInputPolicy", 
        env,
        verbose=1,
        learning_rate=0.0003,
        # Customize network architecture
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        # Tensorboard log needed for WandB sync (WandB will try to find it, or we rely on our callback)
        tensorboard_log=tensorboard_log_dir if PUSH_TO_WANDB else None
    )

    print("Starting training... Press Ctrl+C to stop and save.")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=WandbCallback() 
        )
        print("Training completed normally.")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"ppo_binary_baseline_{timestamp}"
        print(f"Saving model to {save_path}...")
        model.save(save_path)
        print(f"Model saved to {save_path}.zip")
        
        env.close()
        # Close OpenCV windows if any
        if VISUALIZE:
             cv2.destroyAllWindows()
             
        if PUSH_TO_WANDB:
             wandb.finish()
             
        # Cleanup temp logs
        try:
            shutil.rmtree(tensorboard_log_dir)
            print("Temporary TensorBoard logs cleaned up.")
        except:
            pass

if __name__ == "__main__":
    main()
