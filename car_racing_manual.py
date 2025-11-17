import gymnasium as gym
import pygame
import numpy as np


def manual_control():
    pygame.init()
    
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True
    )
    
    while hasattr(env, 'env') and hasattr(env, '_max_episode_steps'):
        env = env.env
    
    print("="*60)
    print("Car Racing - Manual Control")
    print("="*60)
    print("Controls:")
    print("  W - Accelerate")
    print("  A/D - Steer left/right")
    print("  S - Brake")
    print("  ESC or Q - Quit")
    print("="*60)
    print()
    
    observation, info = env.reset()
    action = np.array([0.0, 0.0, 0.0])
    clock = pygame.time.Clock()
    running = True
    total_reward = 0
    step_count = 0
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
            
            keys = pygame.key.get_pressed()
            action = np.array([0.0, 0.0, 0.0])
            
            if keys[pygame.K_a]:
                action[0] = -1.0
            elif keys[pygame.K_d]:
                action[0] = 1.0
            
            if keys[pygame.K_w]:
                action[1] = 1.0
            
            if keys[pygame.K_s]:
                action[2] = 0.8
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"\nEpisode finished - Steps: {step_count}, Reward: {total_reward:.2f}")
                observation, info = env.reset()
                total_reward = 0
                step_count = 0
            
            clock.tick(60)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    manual_control()
