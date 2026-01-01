import gymnasium as gym
import numpy as np
import wandb
import time
import cv2
from dotenv import load_dotenv
from collections import deque

# Script config
FIX_SEED = None # Set to an integer to replay a specific seed (e.g. 12345)
VISUALIZE = False
PUSH_TO_WANDB = True

load_dotenv()

# Speed Constants
TARGET_SPEED = 0.086
# Steering Constants
STEERING_MAGNITUDE = 0.325 # Valid range: [0.0, 1.0]
SENSOR_LOOKAHEAD_Y = 29 # Valid range: [2 (way ahead), 6ß0 (just in front of car)]
SENSOR_OFFSET_X = 7 # Valid range: [2 (close), 12 (wide)]

def get_sensor_coordinates():
    """
    Returns (left_y, left_x, right_y, right_x) based on lookahead and offset.
    Assumes 96x96 observation with car centered at x=48.
    """
    center_x = 48
    ly = SENSOR_LOOKAHEAD_Y
    lx = center_x - SENSOR_OFFSET_X
    ry = SENSOR_LOOKAHEAD_Y
    rx = center_x + SENSOR_OFFSET_X
    return ly, lx, ry, rx

def is_road(pixel):
    r, g, b = pixel
    if int(g) > int(r) + 10 and int(g) > int(b) + 10:
        return False
    return True

def get_action_from_sensors(obs, last_steering):
    h, w, _ = obs.shape
    
    ly_raw, lx_raw, ry_raw, rx_raw = get_sensor_coordinates()
    
    ly = min(max(ly_raw, 0), h-1)
    lx = min(max(lx_raw, 0), w-1)
    ry = min(max(ry_raw, 0), h-1)
    rx = min(max(rx_raw, 0), w-1)
    
    left_pixel = obs[ly, lx]
    right_pixel = obs[ry, rx]
    
    left_is_road = is_road(left_pixel)
    right_is_road = is_road(right_pixel)
    
    steering = 0.0
    gas = TARGET_SPEED
    brake = 0.0
    
    if left_is_road and right_is_road:
        # Straight
        steering = 0.0
    elif left_is_road and not right_is_road:
        # Turn Left
        steering = -STEERING_MAGNITUDE
    elif not left_is_road and right_is_road:
        # Turn Right
        steering = STEERING_MAGNITUDE
    else:
        # Both Grass -> Memory! Maintain last valid turning direction
        # If last steering was straight (0.0), well... we might be lost, but keep straight.
        steering = last_steering
        gas = 0.0 # Stop trying to accelerate if we are completely lost? Or keep pushing? 
        # Actually, for oscillation recovery, usually keeping a little gas helps.
        gas = TARGET_SPEED 
        
    return np.array([steering, gas, brake], dtype=np.float32), left_is_road, right_is_road, steering

import multiprocessing

def run_episode(_):
    """
    Worker function to run a single episode.
    Arguments are ignored but needed for map.
    """
    # Create a fresh environment for each worker
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Get coordinates
    ly_raw, lx_raw, ry_raw, rx_raw = get_sensor_coordinates()
    
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0.0
    
    h, w, _ = obs.shape
    ly = min(max(ly_raw, 0), h-1)
    lx = min(max(lx_raw, 0), w-1)
    ry = min(max(ry_raw, 0), h-1)
    rx = min(max(rx_raw, 0), w-1)
    
    last_steering = 0.0
    
    while not (done or truncated):
        # INLINED LOGIC FOR SPEED
        left_pixel = obs[ly, lx]
        right_pixel = obs[ry, rx]
        
        # is_road logic
        l_road = not (int(left_pixel[1]) > int(left_pixel[0]) + 10 and int(left_pixel[1]) > int(left_pixel[2]) + 10)
        r_road = not (int(right_pixel[1]) > int(right_pixel[0]) + 10 and int(right_pixel[1]) > int(right_pixel[2]) + 10)
        
        steering = 0.0
        gas = TARGET_SPEED
        brake = 0.0
        
        if l_road and r_road:
            steering = 0.0
        elif l_road and not r_road:
            steering = -STEERING_MAGNITUDE
        elif not l_road and r_road:
            steering = STEERING_MAGNITUDE
        else:
            # Memory Case
            steering = last_steering
            
        action = np.array([steering, gas, brake], dtype=np.float32)
        
        # Update memory
        last_steering = steering
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
    env.close()
    return episode_reward

def run_simple_agent(max_episodes=10000):
    # Get coordinates for logging/vis
    ly, lx, ry, rx = get_sensor_coordinates()

    if PUSH_TO_WANDB:
        wandb.init(
            entity="alienpenguin-inc",
            project="rl-training",
            name="visual-heuristic-agent-evaluation",
            config={
                "algorithm": "heuristic-baseline-adaptive",
                "environment": "CarRacing-v3",
                "target_speed": TARGET_SPEED,
                "steering_magnitude": STEERING_MAGNITUDE,
                "sensor_lookahead_y": SENSOR_LOOKAHEAD_Y,
                "sensor_offset_x": SENSOR_OFFSET_X,
                "sensor_left_pos": [ly, lx],
                "sensor_right_pos": [ry, rx],
            }
        )
    
    print(f"Starting Heuristic Baseline Agent on CarRacing-v3 for {max_episodes} episodes...")
    print(f"Target Speed: {TARGET_SPEED}")
    print(f"Steering: {STEERING_MAGNITUDE}")
    print(f"Lookahead Y: {SENSOR_LOOKAHEAD_Y}, Offset X: {SENSOR_OFFSET_X}")
    print(f"Visualization: {'ENABLED' if VISUALIZE else 'DISABLED'}")
    print(f"WandB Logging: {'ENABLED' if PUSH_TO_WANDB else 'DISABLED'}")

    # Track Global Statistics
    all_rewards = []
    
    start_time = time.time()
    try:
        if VISUALIZE:
            # Sequential execution if visualizing
            env = gym.make("CarRacing-v3", render_mode="rgb_array")
            visualize_active = True
            
            for episode in range(max_episodes):
                seed = FIX_SEED if FIX_SEED is not None else None
                obs, _ = env.reset(seed=seed)
                done = False
                truncated = False
                episode_reward = 0.0
                
                last_steering = 0.0
                
                while not (done or truncated):
                    action, l_road, r_road, current_steering = get_action_from_sensors(obs, last_steering)
                    last_steering = current_steering # Update memory
                    
                    if visualize_active:
                        full_frame = env.render()
                        vis_img = full_frame.copy()
                        h_full, w_full, _ = full_frame.shape
                        h_obs, w_obs, _ = obs.shape
                        scale_y = h_full / h_obs
                        scale_x = w_full / w_obs
                        
                        color_road = (0, 255, 0)
                        color_grass = (0, 0, 255)
                        sy_l_raw, sx_l_raw, sy_r_raw, sx_r_raw = get_sensor_coordinates()
                        sy_l = int(sy_l_raw * scale_y)
                        sx_l = int(sx_l_raw * scale_x)
                        c_l = color_road if l_road else color_grass
                        cv2.circle(vis_img, (sx_l, sy_l), 5, c_l, -1)
                        sy_r = int(sy_r_raw * scale_y)
                        sx_r = int(sx_r_raw * scale_x)
                        c_r = color_road if r_road else color_grass
                        cv2.circle(vis_img, (sx_r, sy_r), 5, c_r, -1)
                        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Rule Agent View", vis_img_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Quitting visualization...")
                            visualize_active = False
                            cv2.destroyAllWindows()
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                all_rewards.append(episode_reward)
                global_mean = np.mean(all_rewards)
                
                if PUSH_TO_WANDB:
                    wandb.log({
                        "reward_mean": global_mean,
                        "episode_reward": episode_reward,
                        #"episode": episode + 1
                    })
                print(f"Episode {episode+1}/{max_episodes}: Reward={episode_reward:.2f}, GlobalMean={global_mean:.2f}")
            
            env.close()
            if visualize_active:
                cv2.destroyAllWindows()
                
        else:
            # Parallel execution if NOT visualizing
            num_workers = min(multiprocessing.cpu_count(), 10)  # Use up to 10 cores
            print(f"Running in PARALLEL with {num_workers} workers...")
            
            with multiprocessing.Pool(processes=num_workers) as pool:
                # We use imap_unordered to yield results as they finish
                for i, result in enumerate(pool.imap_unordered(run_episode, range(max_episodes))):
                    episode_reward = result
                    
                    all_rewards.append(episode_reward)
                    global_mean = np.mean(all_rewards)
                    
                    if PUSH_TO_WANDB:
                        wandb.log({
                            "reward_mean": global_mean,
                            "episode_reward": episode_reward,
                            #"episode": i + 1 
                        })
                    
                    # Print every 10 completions to avoid spamming console
                    if (i+1) % 10 == 0:
                         print(f"Episode {i+1}/{max_episodes}: Reward={episode_reward:.2f}, GlobalMean={global_mean:.2f}")
                         
    except KeyboardInterrupt:
        print("\n[STOPPING] Ctrl+C detected. Gracefully finishing...")
    finally:
        elapsed_time = time.time() - start_time
        print(f"\n=== SIMULATION FINISHED ===")
        print(f"Total Time: {elapsed_time:.2f}s")
        if len(all_rewards) > 0:
            final_mean = np.mean(all_rewards)
            final_std = np.std(all_rewards)
            final_median = np.median(all_rewards)
            final_min = np.min(all_rewards)
            final_max = np.max(all_rewards)
            
            print(f"Episodes: {len(all_rewards)}")
            print(f"Mean Reward:   {final_mean:.2f} +/- {final_std:.2f}")
            print(f"Median Reward: {final_median:.2f}")
            print(f"Min / Max:     {final_min:.2f} / {final_max:.2f}")
            
            # Log final stats to WandB summary
            if PUSH_TO_WANDB:
                wandb.run.summary["final_mean_reward"] = final_mean
                wandb.run.summary["final_std_reward"] = final_std
                wandb.run.summary["final_median_reward"] = final_median
                wandb.finish()

if __name__ == "__main__":
    run_simple_agent()
