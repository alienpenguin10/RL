# Reinforcement Learning 

<img src="assets/car_racing.jpeg" alt="Car Racing Environment" width="500">

A collection of reinforcement learning implementations focused on training agents for various environments, including the Car Racing environment.

## 🚀 Setup

### Prerequisites

Use python version between 3.10 to 3.12

Before installation, ensure you have SWIG installed:

**macOS:**
```bash
brew install swig
```

**Windows:**
Follow the instructions at: https://open-box.readthedocs.io/en/latest/installation/install_swig.html

### Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv rl
source rl/bin/activate  # On Windows: rl\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install "ray[rllib]" torch
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your WandB API key from https://wandb.ai/authorize
```

**Note:** The `.env` file contains your WandB API key for experiment tracking. Make sure to add `.env` to your `.gitignore` to keep your API key secure.

## 🎮 Usage

### Car Racing Environment

To get a feel for the environment you can play the game manually:
```bash
python car_racing_manual.py
```
Controls: W (accelerate), A/D (steer), S (brake), ESC/Q (quit)

Run the Car Racing simulation with random actions:
```bash
python car_racing_env.py
```

### Training Examples


**Train PPO Agent on Car Racing:**
```bash
python python run_ppo.py --config ./configs/sac_carracing-throttle-hidden-dims.yaml
```

**Train SAC Agent on Car Racing:**
```bash
python python run_sac.py --config ./configs/sac_carracing-throttle-hidden-dims.yaml
```

## 📂 Project Structure

```
├── 0.Learning/              # Learning materials and basic implementations
│   ├── Deep-Reinforcement-Learning-Notebooks/
│   ├── dqn_cartpole.py
│   ├── q_frozenlake.py
│   └── REINFORCE_lunar_landing.py
├── baseline/            # Baseline implementations for benchmarking
├── agents/                    # Saved models
├── evaluation/                    # Inference and run env in manual mode
├── plots/                  # GIFs and visualizations
├── models/                    # Saved models
├── CarRacingEnv/                    # Environment Adjustments
├── run_sac.py        # Train SAC agent
├── run_vpg_reinforce.py        # Train VPG / REINFORCE agents
└── run_ppo.py        # Train PPO agent
```

## 🧠 Algorithms Implemented

- **REINFORCE**: Monte Carlo policy gradient
- **VPG (Vanilla Policy Gradient)**: Basic policy gradient method
- **PPO (Proximal Policy Optimization)**: State-of-the-art policy gradient
- **SAC (Soft Actor-Critic)**: Off-policy actor-critic algorithm