from agents.networks import ActorCritic, ActorCriticThreeOutput
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

class PPOAgent:
    def __init__(self, device, env,
                 policy_outputs=3, num_epochs=4, batch_size=128,
                 gamma=0.99, gae_lambda=0.95, value_coef=0.01, epsilons=0.2,
                 learning_rate=3e-4, use_lr_scheduler=False, lr_updates=0, entropy_coef=0.02, entropy_decay=1.0,
                 l2_reg=1e-2, max_grad_norm=0.0, process_action_decay=0.0):
        
        self.device = device
        self.observation_dims = env.observation_space.shape
        self.action_dims = policy_outputs
        self.policy = ActorCritic(obs_shape=self.observation_dims, action_dim=2).to(device) if policy_outputs == 2 else ActorCriticThreeOutput(obs_shape=self.observation_dims, action_dim=3).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=lr_updates) if use_lr_scheduler and lr_updates > 0 else None
        self.epochs = num_epochs
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.batch_size = batch_size
        self.epsilons = epsilons
        self.value_coeff = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.process_action_coef = 1.0 if process_action_decay > 0.0 else 0.0
        self.max_grad_norm = max_grad_norm
        self.process_action_decay = process_action_decay
    
    def update(self, rollouts):
        # rollouts: {'states': states, 'actions': actions, 'returns': returns, 'advantages': advantages, 'values': values, 'log_probs': log_probs}
        states = rollouts['states']
        actions = rollouts['actions'] 
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        values = rollouts['values']
        old_log_probs = rollouts['log_probs']

        # Normalize advantages
        normalised_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, normalised_advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch
                log_probs, values, entropy = self.policy.evaluate(b_states, b_actions)
                
                # Policy Loss Formula
                # r = log(pi(a|s)) / log_old(pi(a|s))
                # L_clip = min(r * A, clip(r, 1-eps, 1+eps) * A)
                # Policy Loss = - E[L_clip]
                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilons, 1 + self.epsilons) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss Formula
                # Value Loss = 1/2 E[(V(s) - V(s'))^2]
                b_returns = b_returns.reshape(-1)
                b_values = values.reshape(-1)
                value_loss = 0.5 * (b_returns - b_values).pow(2).mean()

                # Policy Entropy = E[log(pi(a|s))]
                entropy_loss = entropy.mean()

                # Total Loss = Policy Loss - ENTROPY_COEFF * Policy Entropy + VALUE_COEFF * Value Loss
                total_loss = policy_loss - self.entropy_coef * entropy_loss + self.value_coeff * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Decay entropy coefficient
        if self.entropy_decay < 1.0:
            self.entropy_coef *= self.entropy_decay
        # Decay action processing influence
        if self.process_action_decay < 1.0:
            self.process_action_coef *= self.process_action_decay
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
        }

    def compute_gae(self, rewards, values, terminateds, truncateds, next_value):
        #TD Error = r + gamma * V(s_{t+1}) - V(s_t)
        # A_t = TD Error + gamma * lambda * A(s_{t+1})
        # Recall returns can be computed in two different ways:
        # 1. Monte Carlo returns: G_t = gamma^k * r_t + gamma^(k-1) * r_(t+1) + ... + gamma * r_(t+k-1)
        # 2. GAE returns: G_t = A_t + V(s_t) since A_t = G_t - V(s_t)
        # returns: Uses returns as targets to train the critic function to predict better state, value predictions.
        # terminated: bootstrap_mask=0, gae_mask=0
        # truncated: bootstrap_mask=1, gae_mask=1
        # (i.e., we DO bootstrap and accumulate for truncated episodes)
        advantages = [] # Uses this to determine which actions were better than expected, helping the policy improve.
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # This is the last step of rollout; Only mask terminated states, not truncated ones
                next_non_terminal = 1.0 - terminateds[t]
                next_values = next_value
            else:
                # Use the NEXT transition's terminated flag
                next_non_terminal = 1.0 - terminateds[t + 1]
                next_values = values[t + 1]

            # TD error with proper masking
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            
            # GAE accumulation
            # If terminated (episode ended naturally), don't accumulate future advantages
            # If truncated (episode ended due to time limit),DO accumulate (we bootstrapped above)
            terminated_mask = terminateds[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - terminated_mask) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return torch.tensor(returns).to(self.device), torch.tensor(advantages).to(self.device)

    def process_action(self, action, info=None, steering_buffer=None):
        if info is None or len(info) == 0:
            return action

        processed_action = action.copy()
        speed = info.get("speed")
        if speed is not None and self.process_action_coef > 0.0:
            # Limit braking based on current speed to encourage forward movement
            if self.action_dims == 2:
                if speed < 20.0:
                    processed_action[1] = max(action[1], 0.1) # Must move forward at low speeds
                elif speed < 25.0:
                    processed_action[1] = max(action[1], 0.0) # No braking at low speeds
                elif speed < 40.0:
                    processed_action[1] = max(action[1], -0.05) # Limit braking at moderate speeds

            # Influence of processing can decay over time
            if self.process_action_coef < 1.0:
                processed_action[1] = action[1] + (processed_action[1] - action[1]) * self.process_action_coef

        # if steering_buffer is not None:
        #     # Limit steering input
        #     avg_steer = np.mean(steering_buffer) if len(steering_buffer) > 0 else 0.0
        #     steer = 0.7 * steer + 0.3 * avg_steer
        #     steer = np.clip(steer, -0.7, 0.7)

        # print(f"Processed action - Steer: {processed_action[0]}, Speed: {processed_action[1]}")
        
        return processed_action
    
    def save_model(self, filepath):
        """
        Saves the model to a file
        """
        save_dict = {}
        if hasattr(self, 'policy'):
            save_dict['policy'] = self.policy.state_dict()
        torch.save(save_dict, filepath)

    def load_model(self, filepath):
        """
        Loads the model from a file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        if hasattr(self, 'policy') and 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'])
