# REINFORCE with Value Function Baseline

This is an implementation of the REINFORCE algorithm with a value function baseline (Actor-Critic style) for continuous control tasks, specifically designed for the CarRacing-v3 environment.

## Overview

REINFORCE is a policy gradient algorithm that learns a policy by directly optimizing the expected return. This implementation uses:

- **Policy Gradient**: Updates the policy using the gradient of expected return
- **Value Function Baseline**: Reduces variance by subtracting a value function estimate from returns
- **Advantage Normalization**: Normalizes advantages to stabilize training
- **Per-Episode Updates**: Updates the policy after each episode

## Algorithm

The REINFORCE algorithm with baseline works as follows:

1. **Collect Episode**: Run the current policy for one episode, storing:
   - States: `s_0, s_1, ..., s_{T-1}`
   - Actions: `a_0, a_1, ..., a_{T-1}`
   - Rewards: `r_0, r_1, ..., r_{T-1}`
   - Log probabilities: `log π(a_t | s_t)`
   - Value estimates: `V(s_t)`

2. **Compute Returns**: Calculate discounted returns for each timestep:
   ```
   G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t-1}·r_{T-1}
   ```

3. **Compute Advantages**: Subtract value estimates from returns:
   ```
   A(s_t, a_t) = G_t - V(s_t)
   ```
   This tells us how much better (or worse) the action was compared to the expected value.

4. **Normalize Advantages**: Normalize to mean 0 and std 1 for stability:
   ```
   A_normalized = (A - mean(A)) / (std(A) + ε)
   ```

5. **Update Policy**: Maximize the expected advantage-weighted log probability:
   ```
   L_policy = -mean(log π(a_t | s_t) × A_normalized)
   ```
   - Positive advantage → increase action probability
   - Negative advantage → decrease action probability

6. **Update Value Function**: Fit value estimates to actual returns:
   ```
   L_value = mean((V(s_t) - G_t)²)
   ```

## Files

- **`networks.py`**: Neural network architectures
  - `PolicyNetwork`: CNN-based policy network for image observations
    - Outputs mean and log_std for each action dimension (steering, gas, brake)
    - Uses Gaussian distributions for continuous actions
  - `ValueNetwork`: CNN-based value function network
    - Estimates state values V(s)

- **`agents.py`**: REINFORCE agent implementation
  - `REINFORCEAgent`: Main agent class
    - `select_action()`: Sample action from policy
    - `store_transition()`: Store state, action, reward
    - `update()`: Perform policy and value function updates

- **`utils.py`**: Utility functions
  - `compute_returns()`: Calculate discounted returns
  - `compute_advantages()`: Compute advantages using baseline
  - `normalize_advantages()`: Normalize advantages for stability

- **`train.py`**: Training script
  - Main training loop
  - Episode collection and updates
  - Reward tracking and plotting

## Usage

```bash
cd Reinforce
python train.py
```

## Hyperparameters

Default hyperparameters:

- `learning_rate = 0.0003`: Learning rate for both policy and value networks
- `discount_factor = 0.99`: Discount factor (γ) for future rewards
- `num_episodes = 2500`: Number of training episodes
- `max_ep_len = 1000`: Maximum episode length (handled by environment)

## Network Architecture

### Policy Network
- **Input**: 96×96×3 RGB images (CarRacing-v3 observations)
- **CNN Layers**:
  - Conv1: 32 filters, 8×8 kernel, stride 4 → 23×23×32
  - Conv2: 64 filters, 4×4 kernel, stride 2 → 10×10×64
  - Conv3: 64 filters, 3×3 kernel, stride 1 → 8×8×64
- **Fully Connected**:
  - FC1: 4096 → 512
  - FC2: 512 → 64
- **Output Heads** (separate for each action dimension):
  - Steering: mean (tanh), log_std → [-1, 1]
  - Gas: mean (sigmoid), log_std → [0, 1]
  - Brake: mean (sigmoid), log_std → [0, 1]

### Value Network
- Same CNN architecture as policy network
- Final output: Single scalar value estimate V(s)

## Key Features

1. **Value Function Baseline**: Reduces variance compared to vanilla REINFORCE
2. **Advantage Normalization**: Stabilizes training by normalizing advantages
3. **Gradient Clipping**: Prevents exploding gradients (max_norm=0.5)
4. **Continuous Actions**: Handles continuous action spaces with Gaussian policies
5. **Image Observations**: Uses CNN to process pixel observations

## Comparison with VPG

| Feature | REINFORCE | VPG |
|---------|-----------|-----|
| **Advantage** | Simple: A = G - V | GAE-Lambda (λ=0.97) |
| **Update Frequency** | After each episode | After each epoch (4000 steps) |
| **Value Training** | 1 update per episode | 80 updates per epoch |
| **Collection** | Single episode | Multiple trajectories |
| **Advantage Normalization** | Yes | No (typically) |

## Training Output

The training script:
- Prints episode statistics (reward, steps, losses)
- Saves models every 10 episodes to `models/model_{episode}.pth`
- Generates reward plots with moving averages
- Saves final model and plot at the end

## Model Saving

Models are saved with:
- Policy network state dict
- Value network state dict
- Policy optimizer state dict
- Value optimizer state dict

## Environment

Designed for **CarRacing-v3** from Gymnasium:
- **Observation Space**: 96×96×3 RGB images
- **Action Space**: Continuous 3D (steering [-1,1], gas [0,1], brake [0,1])
- **Reward**: Sparse rewards based on track completion

## Future Improvements (TODOs)

- Frame stacking: Stack 4 consecutive frames to capture motion
- Grayscale conversion: Reduce computation by using grayscale
- Frame skipping: Act every 4 frames for faster training
- Reward shaping: Modify sparse rewards for better learning signal

## References

- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
- Policy Gradient Methods: Direct optimization of policy parameters using gradient ascent

