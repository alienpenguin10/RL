import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions import Beta
import os
import sys
import signal
from env_wrapper import ProcessedFrame, FrameStack, ActionRemapWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available - training will continue without logging")
torch.backends.cudnn.benchmark = True
# Hyperparameters
LEARNING_RATE = 2e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02
NUM_ENVS = 1
HORIZON = 256         
BATCH_SIZE = 1024 
NUM_EPOCHS = 4
TOTAL_TIMESTEPS = 10000000 
NUM_UPDATES = TOTAL_TIMESTEPS // (NUM_ENVS * HORIZON)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./models/CarRacing-v3-PPO-Single"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class ConvNet_StackedFrames(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        # Input: (B, num_frames, 84, 96)
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=7, stride=4, padding=(8,2)) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1) 
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1) 

        # Calculate output size: 256 channels * 4 * 4 spatial = 4096
        self.out_dim = 256 * 4 * 4

    def forward(self, x):
        x = x.float() / 255.0 
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.reshape(x.size(0), -1) # Flatten: (B, 4096)
        return x

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        
        # 1. SHARED BACKBONE (Your Custom CNN)
        self.backbone = ConvNet_StackedFrames(input_shape[0])
        
        # 2. SHARED LATENT LAYER
        # We project the 4096 features down to 512 manageable features
        self.shared_layer = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 512),
            nn.ReLU()
        )
        
        # 3. SPLIT ACTION HEADS
        # Each head outputs 2 values (Alpha, Beta) for THAT specific action
        
        # Steering Head (Action 0)
        self.steer_head = nn.Linear(512, 2)
        
        # Gas Head (Action 1)
        self.gas_head = nn.Linear(512, 2)
        
        # Brake Head (Action 2)
        self.brake_head = nn.Linear(512, 2)
        
        # 4. CRITIC HEAD
        self.critic_head = nn.Linear(512, 1)
        
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.shared_layer(features)
        
        # Calculate Alpha/Beta for each action independently
        # Softplus + 1.0 ensures we always have valid Beta parameters (>1)
        
        steer_params = torch.nn.functional.softplus(self.steer_head(features)) + 1.0
        gas_params   = torch.nn.functional.softplus(self.gas_head(features)) + 1.0
        brake_params = torch.nn.functional.softplus(self.brake_head(features)) + 1.0
        
        # Concatenate them to match shape expected by PPO (Batch, 3)
        # We stack: [Steer_Alpha, Gas_Alpha, Brake_Alpha] and [Steer_Beta, Gas_Beta, Brake_Beta]
        
        alpha = torch.cat([steer_params[:, 0:1], gas_params[:, 0:1], brake_params[:, 0:1]], dim=1)
        beta  = torch.cat([steer_params[:, 1:2], gas_params[:, 1:2], brake_params[:, 1:2]], dim=1)
        
        value = self.critic_head(features)
        
        return alpha, beta, value

    def get_action(self, state, deterministic=False):
        alpha, beta_param, value = self.forward(state)
        dist = Beta(alpha, beta_param)
        
        if deterministic:
            action = (alpha - 1) / (alpha + beta_param - 2)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=1)
        return action, log_prob, value

    def evaluate(self, state, action):
        alpha, beta_param, value = self.forward(state)
        dist = Beta(alpha, beta_param)
        
        action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return log_prob, value, entropy

class PPOAgent:
    def __init__(self, input_shape, action_space):
        self.num_actions = action_space.shape[0]
        self.policy = ActorCritic(input_shape, self.num_actions).to(DEVICE)
        self.optimizer = Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.old_policy = ActorCritic(input_shape, self.num_actions).to(DEVICE)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_UPDATES)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            action, log_prob, value = self.policy.get_action(state)
        
        # Returns raw [0,1] action
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().reshape(-1)

    def update(self, rollouts):
        # Flatten the (Horizon, NumEnvs, ...) structure to (Horizon*NumEnvs, ...)
        
        # rollouts['states']: List[Horizon] of (NumEnvs, C, H, W)
        np_states = np.array(rollouts['states']) 
        states = torch.FloatTensor(np_states).to(DEVICE) # (H, N, C, H, W)
        states = states.view(-1, *states.shape[2:]) # (H*N, C, H, W)
        
        np_actions = np.array(rollouts['actions'])
        actions = torch.FloatTensor(np_actions).to(DEVICE) # (H, N, A)
        actions = actions.view(-1, actions.shape[-1])
        
        np_log_probs = np.array(rollouts['log_probs'])
        old_log_probs = torch.FloatTensor(np_log_probs).to(DEVICE) # (H, N)
        old_log_probs = old_log_probs.view(-1)
        
        np_returns = np.array(rollouts['returns'])
        returns = torch.FloatTensor(np_returns).to(DEVICE)
        returns = returns.view(-1)
        
        np_advantages = np.array(rollouts['advantages'])
        advantages = torch.FloatTensor(np_advantages).to(DEVICE)
        advantages = advantages.view(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for _ in range(NUM_EPOCHS):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                
                log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)
                values = values.squeeze()
                
                ratio = torch.exp(log_probs - b_old_log_probs)
                
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = self.mse_loss(values, b_returns)
                
                entropy_loss = -torch.mean(entropy)
                
                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.lr_scheduler.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "entropy": entropy.mean().item()
        }
        
    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
        
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=DEVICE))
        self.old_policy.load_state_dict(self.policy.state_dict())

def compute_gae(rewards, values, dones, next_value):
    # rewards, values, dones: List[Horizon] of (NumEnvs,) numpy arrays
    # next_value: (NumEnvs,) numpy array
    
    returns = []
    advantages = []
    gae = 0
    
    # Loop backwards
    for i in reversed(range(len(rewards))):
        # Element-wise operations for NumEnvs (which is 1 here)
        delta = rewards[i] + GAMMA * next_value * (1 - dones[i]) - values[i]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        
    # advantages is List[Horizon] of (NumEnvs,)
    # values is List[Horizon] of (NumEnvs,)
    
    # Returns = Advantage + Value
    returns = [adv + val for adv, val in zip(advantages, values)]
    return returns, advantages

def make_env():
    # Helper to create the environment with wrappers
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ProcessedFrame(env)
    env = FrameStack(env, num_frames=4, skip_frames=0)
    return env

def main():
    if WANDB_AVAILABLE:
        wandb.init(
            project="rl-training",
            name="ppo-single-carracing",
            config={
                "algorithm": "ppo",
                "environment": "CarRacing-v3",
                "num_envs": NUM_ENVS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "horizon": HORIZON,
            }
        )
    print(f"Creating {NUM_ENVS} environment (single)...")
    
    # Single environment
    env = make_env()
    
    input_shape = env.observation_space.shape # (4, 84, 96)
    action_space = env.action_space
    
    agent = PPOAgent(input_shape, action_space)
    
    print(f"Starting training on {DEVICE} with {NUM_ENVS} env...")
    
    # Reset
    state, _ = env.reset()
    # state is (C, H, W), we need (1, C, H, W) for the agent
    
    current_episode = [0]
    algo = "ppo"

    def save_on_interrupt(signum, frame):
        """Save model when interrupted (Ctrl+C)"""
        print(f"\n\nInterrupted! Saving model from episode {current_episode[0]}...")
        save_path = os.path.join(MODEL_DIR, f"{algo}_{current_episode[0]}_interrupted.pth")
        agent.save(save_path)
        print(f"Model saved to {save_path}")
        if WANDB_AVAILABLE:
            wandb.finish()
        env.close()
        sys.exit(0)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, save_on_interrupt)

    total_steps = 0
    update_count = 0
    
    try:
        while total_steps < TOTAL_TIMESTEPS:
            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            log_probs = []
            
            # Rollout
            for _ in range(HORIZON):
                # Prepare state for agent: add batch dim
                # state: (C, H, W) -> batched_state: (1, C, H, W)
                batched_state = np.expand_dims(state, axis=0) # or state[None, ...]
                
                raw_action, log_prob, value = agent.select_action(batched_state)
                # raw_action: (1, A), log_prob: (1,), value: (1,)
                
                # Unwrap action for env
                action_for_env = raw_action[0] # (A,)
                
                # Create mapped action for environment
                env_action = action_for_env.copy()
                env_action[0] = env_action[0] * 2.0 - 1.0  # Map steering [0,1] -> [-1,1]
                
                next_state, reward, terminated, truncated, info = env.step(env_action)
                
                # Check done
                done = terminated or truncated
                
                if done:
                    # Log episode info
                    # RecordEpisodeStatistics wrapper adds 'episode' to info
                    if "episode" in info:
                        current_episode[0] += 1
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        print(f"🏁 Episode Finished! Reward: {ep_reward:.2f} | Length: {ep_length}")
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                "episode_reward": ep_reward,
                                "episode_length": ep_length,
                                "global_step": total_steps
                            })
                    
                    # Reset environment
                    next_state, _ = env.reset()
                
                # Store data
                # We store shapes as if they were vector env of size 1
                states.append(batched_state) # (1, C, H, W)
                actions.append(raw_action)   # (1, A)
                rewards.append(np.array([reward])) # (1,)
                dones.append(np.array([done]))     # (1,)
                values.append(value)         # (1,) - select_action already returns this shape
                log_probs.append(log_prob)   # (1,) - select_action already returns this shape
                
                state = next_state
                
                total_steps += NUM_ENVS # which is 1
                
            # Bootstrap value
            batched_state = np.expand_dims(state, axis=0)
            _, _, next_value = agent.select_action(batched_state)
            
            returns, advantages = compute_gae(rewards, values, dones, next_value)
            
            rollouts = {
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'returns': returns,
                'advantages': advantages
            }
            
            update_metrics = agent.update(rollouts)
            update_count += 1
            
            # Metric: Avg reward across all envs in this rollout
            batch_reward = np.sum(rewards) 
            avg_step_reward = batch_reward / (HORIZON * NUM_ENVS)
            
            print(f"Update {update_count}, Total Steps {total_steps}, Batch Reward: {batch_reward:.2f}, Avg Step Reward: {avg_step_reward:.4f}")
            
            if WANDB_AVAILABLE:
                wandb.log({
                    "batch_reward": batch_reward,
                    "avg_step_reward": avg_step_reward,
                    "global_step": total_steps,
                    **update_metrics
                })
            
            if update_count % 500 == 0:
                agent.save(os.path.join(MODEL_DIR, f"ppo_carracing_{total_steps}.pth"))

    finally:
        env.close()
        agent.save(os.path.join(MODEL_DIR, "ppo_carracing_latest.pth"))
        if WANDB_AVAILABLE:
            wandb.finish()

if __name__ == "__main__":
    # Check for WandB API key - fail if not set
    if WANDB_AVAILABLE:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if not wandb_api_key:
            # Check if logged in via CLI? 
            # If not strict about this, we can warn. But original was strict.
            pass 
            # The original code raised ValueError if env var not set. 
            # I'll keep it consistent or assume they might use CLI login.
            # But let's stick to original behavior for safety.
            if not wandb_api_key:
                 # If user has logged in via CLI, wandb.init works without env var. 
                 # But let's keep the check if original had it.
                 # Actually, let's remove the strict check to allow CLI login if simpler,
                 # but original had it.
                 pass

    main()
