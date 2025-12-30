import pygame
import numpy as np
import cv2 as cv

episode = 2
reward = -118
recording_path = f"./models/replay/ppo_replay_ep{episode}_reward{reward}.npz"
recording = np.load(recording_path)
states = recording['states']
print("Loaded recording with", len(states), "states.")

pygame.init()
screen_size = (800, 700)
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

running = True
i = 0

while running and i < len(states):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False

    state_frame = states[i][-1] # Use the last frame in the stacked frames
    state_display = (state_frame * 255).astype(np.uint8)  # Scale to [0, 255]
    state_display = cv.resize(state_display, screen_size)  # Resize for better visibility

    # Convert single channel to RGB grayscale for pygame
    state_display = np.stack([state_display] * 3, axis=-1)
    state_display = state_display.transpose(1, 0, 2)  # Transpose for pygame (swap width/height)
    surface = pygame.surfarray.make_surface(state_display)
    screen.blit(surface, (0, 0))

    pygame.display.flip()
    clock.tick(30) # Limit to 30 FPS
    i += 1

pygame.quit()