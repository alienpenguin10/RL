import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions import Beta
from env_wrapper import ProcessedFrame, FrameStack # Assuming these are in your env_wrapper.py

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "./models/CarRacing-v3-PPO-Vector/ppo_carracing_4096000.pth"
VIDEO_FOLDER = "./recorded_videos"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 3  # How many laps to record

# ==========================================
# 2. ARCHITECTURE (Must Match Training Exactly)
# ==========================================
class ConvNet_StackedFrames(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=7, stride=4, padding=(8,2)) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1) 
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1) 
        self.out_dim = 256 * 4 * 4

    def forward(self, x):
        x = x.float() / 255.0 
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.reshape(x.size(0), -1)
        return x

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        self.backbone = ConvNet_StackedFrames(input_shape[0])
        self.shared_layer = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 512),
            nn.ReLU()
        )
        self.steer_head = nn.Linear(512, 2)
        self.gas_head = nn.Linear(512, 2)
        self.brake_head = nn.Linear(512, 2)
        self.critic_head = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.shared_layer(features)
        
        steer_params = torch.nn.functional.softplus(self.steer_head(features)) + 1.0
        gas_params   = torch.nn.functional.softplus(self.gas_head(features)) + 1.0
        brake_params = torch.nn.functional.softplus(self.brake_head(features)) + 1.0
        
        alpha = torch.cat([steer_params[:, 0:1], gas_params[:, 0:1], brake_params[:, 0:1]], dim=1)
        beta  = torch.cat([steer_params[:, 1:2], gas_params[:, 1:2], brake_params[:, 1:2]], dim=1)
        value = self.critic_head(features)
        return alpha, beta, value

    def get_action(self, state, deterministic=False):
        alpha, beta_param, value = self.forward(state)
        dist = Beta(alpha, beta_param)
        if deterministic:
            # Use the mode (most likely action) for evaluation
            action = (alpha - 1) / (alpha + beta_param - 2)
        else:
            action = dist.sample()
        return action, None, value

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def record_agent():
    print(f"Loading model from: {MODEL_PATH}")
    
    # Create Environment
    # render_mode="rgb_array" is required for recording on servers
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    
    # Wrap for recording
    # We set episode_trigger to lambda x: True so it records EVERY episode we run here
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=VIDEO_FOLDER, 
        episode_trigger=lambda x: True,
        name_prefix="eval-run"
    )
    
    # Apply standard preprocessing wrappers
    env = ProcessedFrame(env)
    env = FrameStack(env, num_frames=4, skip_frames=0)

    # Initialize Model
    input_shape = (4, 84, 96)
    num_actions = 3
    model = ActorCritic(input_shape, num_actions).to(DEVICE)
    
    # Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Starting recording for {NUM_EPISODES} episodes...")
    print(f"Videos will be saved to: {os.path.abspath(VIDEO_FOLDER)}")

    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # Prepare state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Get deterministic action
                action, _, _ = model.get_action(state_tensor, deterministic=True)
            
            # Action Mapping (CRITICAL)
            raw_action = action.cpu().numpy()[0]
            env_action = raw_action.copy()
            
            # Map Steering: [0, 1] -> [-1, 1]
            env_action[0] = env_action[0] * 2.0 - 1.0
            # Gas/Brake: [0, 1] -> [0, 1] (No change needed)

            state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
        print(f"Episode {ep+1}: Score {total_reward:.2f} | Steps {step}")

    env.close()
    print("Done! Download the video files from the 'recorded_videos' folder.")

if __name__ == "__main__":
    record_agent()