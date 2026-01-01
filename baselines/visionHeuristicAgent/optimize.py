import gymnasium as gym
import numpy as np
import time
import cv2  # Just for processing/debugging if needed
from collections import deque
import multiprocessing
import wandb
from dotenv import load_dotenv
import os

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
# ------------------------------------------------------------------------------
# OBJECTIVE FUNCTION TO OPTIMIZE
# ------------------------------------------------------------------------------

def evaluate_params_wrapper(args):
    """
    Wrapper to unpack arguments for pool.map.
    args: (params, num_episodes)
    """
    params, num_episodes = args
    return evaluate_params(params, num_episodes)

def evaluate_params(params, num_episodes=5):
    """
    Runs MULTIPLE episodes with the given parameters and returns the AVERAGE Reward
    AND the seed of the best episode in this batch.
    params = (target_speed, steering_magnitude, lookahead_y, offset_x)
    Returns: (avg_score, best_episode_seed, best_episode_score)
    """
    target_speed, steering_mag, lookahead_y, offset_x = params
    
    # Clip parameters to valid ranges
    target_speed = np.clip(target_speed, 0.05, 1.0)
    steering_mag = np.clip(steering_mag, 0.05, 1.0)
    lookahead_y = int(np.clip(lookahead_y, 0, 84))
    offset_x = int(np.clip(offset_x, 2, 45))
    
    total_score = 0.0
    best_ep_score = -float('inf')
    best_ep_seed = None
    
    for i in range(num_episodes):
        # Generate a random seed
        seed = np.random.randint(0, 1000000)
        
        env = gym.make("CarRacing-v3")
        obs, _ = env.reset(seed=seed)
        
        episode_reward = 0.0
        done = False
        truncated = False
        
        # Calculate sensor positions
        center_x = 48
        ly = lookahead_y
        lx = center_x - offset_x
        ry = lookahead_y
        rx = center_x + offset_x
        
        # Pre-clamp sensor coordinates
        ly = min(max(ly, 0), 95)
        lx = min(max(lx, 0), 95)
        ry = min(max(ry, 0), 95)
        rx = min(max(rx, 0), 95)
        
        steps = 0
        max_steps = 1000  # Cap episode length for speed
        
        last_steering = 0.0
        
        while not (done or truncated) and steps < max_steps:
            # --- AGENT LOGIC INLINED FOR SPEED ---
            left_pixel = obs[ly, lx]
            right_pixel = obs[ry, rx]
            
            # is_road logic
            l_road = not (int(left_pixel[1]) > int(left_pixel[0]) + 10 and int(left_pixel[1]) > int(left_pixel[2]) + 10)
            r_road = not (int(right_pixel[1]) > int(right_pixel[0]) + 10 and int(right_pixel[1]) > int(right_pixel[2]) + 10)
            
            steering = 0.0
            gas = target_speed
            
            if l_road and r_road:
                steering = 0.0
            elif l_road and not r_road:
                steering = -steering_mag
            elif not l_road and r_road:
                steering = steering_mag
            else:
                 # Memory case
                steering = last_steering
                
            action = np.array([steering, gas, 0.0], dtype=np.float32)
            last_steering = steering
            # -------------------------------------
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
        env.close()
        total_score += episode_reward
        
        if episode_reward > best_ep_score:
            best_ep_score = episode_reward
            best_ep_seed = seed

    return total_score / num_episodes, best_ep_seed, best_ep_score

# ------------------------------------------------------------------------------
# GENETIC ALGORITHM
# ------------------------------------------------------------------------------

def run_optimization():
    # POPULATION CONFIG
    POP_SIZE = 50
    GENERATIONS = 20
    ELITE_SIZE = 5
    MUTATION_RATE = 0.2
    MUTATION_SCALE = 0.1
    
    # Params: [TargetSpeed, Steering, LookaheadY, OffsetX]
    # Initial guessing ranges
    pop = []
    for _ in range(POP_SIZE):
        ind = [
            np.random.uniform(0.05, 0.15),# Target Speed (slower than adaptive max)
            np.random.uniform(0.1, 0.5),  # Steering
            np.random.randint(20, 70),    # Lookahead Y
            np.random.randint(5, 20)      # Offset X
        ]
        pop.append(ind)
        
    pop = np.array(pop)
    
    print(f"Starting Genetic Algorithm Optimization...")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}")
    print(f"Workers: {multiprocessing.cpu_count()}")
    
    # Init WandB for Optimization
    wandb.init(
        entity="alienpenguin-inc",
        project="rl-training",
        name="visual-heuristic-agent-optimization",
        config={
            "pop_size": POP_SIZE,
            "generations": GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "params": "TargetSpeed, Steering, Y, X"
        }
    )
    
    best_overall_score = -float('inf')
    best_overall_ind = None
    best_overall_seed = None # Track the seed that got the high score
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    try:
        for gen in range(GENERATIONS):
            start_time = time.time()
            
            # Adaptive difficulty: More episodes for later generations to really refine the best
            if gen >= 15:
                num_episodes = 10
            else:
                num_episodes = 5
                
            print(f"Gen {gen+1}: Training with {num_episodes} eps/ind...")
            
            # 1. EVALUATE
            # Create args list for wrapper
            eval_args = [(ind, num_episodes) for ind in pop]
            results = pool.map(evaluate_params_wrapper, eval_args)
            
            # Unpack results
            # results is list of (avg_score, best_seed, best_ep_score)
            scores = np.array([r[0] for r in results])
            seeds = [r[1] for r in results]
            ep_scores = [r[2] for r in results]
            
            # 2. SELECT BEST
            best_idx = np.argsort(scores)[-ELITE_SIZE:]
            elites = pop[best_idx]
            best_avg_score = scores[best_idx[-1]] # Best AVG score in THIS generation
            avg_score = np.mean(scores) # Average score of population
            
            current_best_ind = elites[-1]
            
            # Update BEST EVER FOUND
            if best_avg_score > best_overall_score:
                best_overall_score = best_avg_score
                best_overall_ind = current_best_ind.copy()
                # We save the seed of the best single episode from this best agent for replay
                # Find which episode for this agent was best
                best_agent_idx = best_idx[-1]
                best_overall_seed = seeds[best_agent_idx]
                print(f"  > New Best Avg: {best_overall_score:.2f} (Best Ep Seed: {best_overall_seed})")
            
            print(f"Gen {gen+1}: BestAvg={best_avg_score:.2f}, PopAvg={avg_score:.2f} | Params: Spd={current_best_ind[0]:.3f}, Steer={current_best_ind[1]:.3f}, Y={int(current_best_ind[2])}, X={int(current_best_ind[3])} | Time: {time.time()-start_time:.2f}s")
            
            wandb.log({
                # "generation": gen + 1,
                # "gen_best_avg_score": best_avg_score,
                # "global_best_avg_score": best_overall_score,
                "pop_avg_score": avg_score,
                "best_target_speed": current_best_ind[0],
                "best_steering": current_best_ind[1],
                "best_lookahead_y": current_best_ind[2],
                "best_offset_x": current_best_ind[3]
            })

            # 3. CREATE NEXT GEN
            new_pop = []
            new_pop.extend(elites) # Elitism
            
            while len(new_pop) < POP_SIZE:
                # Tournament Selection
                p1_idx = np.random.randint(0, POP_SIZE)
                p2_idx = np.random.randint(0, POP_SIZE)
                parent1 = pop[p1_idx] if scores[p1_idx] > scores[p2_idx] else pop[p2_idx]
                
                p3_idx = np.random.randint(0, POP_SIZE)
                p4_idx = np.random.randint(0, POP_SIZE)
                parent2 = pop[p3_idx] if scores[p3_idx] > scores[p4_idx] else pop[p4_idx]
                
                # Crossover
                child = (parent1 + parent2) / 2.0
                
                # Mutation
                if np.random.random() < MUTATION_RATE:
                    # Mutate one gene
                    gene_idx = np.random.randint(0, 4) # 0 to 3
                    child[gene_idx] += np.random.normal(0, MUTATION_SCALE)
                
                new_pop.append(child)
                
            pop = np.array(new_pop)
            
    except KeyboardInterrupt:
        print("\n[STOPPING] Ctrl+C detected. Gracefully finishing...")
        
    print(f"\nOPTIMIZATION COMPLETE")
    print(f"Best Agent Avg Score: {best_overall_score:.2f}")
    if best_overall_seed is not None:
        print(f"Seed for Best Episode of Champion: {best_overall_seed}")
    
    if best_overall_ind is not None:
        print(f"Best Params:")
        print(f"  Target Speed: {best_overall_ind[0]:.3f}")
        print(f"  Steering: {best_overall_ind[1]:.3f}")
        print(f"  Lookahead Y: {int(best_overall_ind[2])}")
        print(f"  Offset X: {int(best_overall_ind[3])}")
    
    wandb.finish()
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    run_optimization()
