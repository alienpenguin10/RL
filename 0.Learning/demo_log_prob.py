"""
Demonstration of how log_prob(actions).sum(dim=1) transforms values
"""
import torch
import torch.nn as nn

# Simulate a batch of 3 states and their corresponding actions
batch_size = 3
action_dim = 3  # steering, gas, brake

print("=" * 60)
print("DEMONSTRATION: log_prob(actions).sum(dim=1) Transformation")
print("=" * 60)

# Step 1: Example actions that were taken (batch_size=3, action_dim=3)
# Each row is one action: [steering, gas, brake]
actions = torch.tensor([
    [0.5, 0.8, 0.2],   # Action 1: slight right turn, high gas, low brake
    [-0.3, 0.4, 0.6],  # Action 2: left turn, medium gas, medium brake
    [0.0, 0.9, 0.1]    # Action 3: straight, high gas, low brake
], dtype=torch.float32)

print(f"\n1. INPUT ACTIONS (shape: {actions.shape})")
print("   Each row = [steering, gas, brake] for one state")
print(actions)
print(f"   Shape: {actions.shape} = [batch_size={batch_size}, action_dim={action_dim}]")

# Step 2: Example means and stds from the policy network
# These would come from self.forward(states) in the actual code
means = torch.tensor([
    [0.4, 0.7, 0.15],   # Policy's predicted mean for state 1
    [-0.2, 0.5, 0.5],   # Policy's predicted mean for state 2
    [0.1, 0.85, 0.12]   # Policy's predicted mean for state 3
], dtype=torch.float32)

stds = torch.tensor([
    [0.2, 0.1, 0.05],   # Policy's predicted std for state 1
    [0.15, 0.12, 0.08], # Policy's predicted std for state 2
    [0.18, 0.08, 0.06]  # Policy's predicted std for state 3
], dtype=torch.float32)

print(f"\n2. POLICY DISTRIBUTION PARAMETERS")
print(f"   MEANS (shape: {means.shape})")
print(means)
print(f"   STDS (shape: {stds.shape})")
print(stds)

# Step 3: Create Normal distribution
dist = torch.distributions.Normal(means, stds)

# Step 4: Compute log_prob for each action component
log_probs_per_component = dist.log_prob(actions)

print(f"\n3. LOG PROBABILITIES PER COMPONENT (shape: {log_probs_per_component.shape})")
print("   Each value = log P(action_component | state)")
print("   Shape: [batch_size=3, action_dim=3]")
print(log_probs_per_component)
print("\n   Breaking it down:")
print("   Row 0: [log_prob(steering=0.5), log_prob(gas=0.8), log_prob(brake=0.2)]")
print("   Row 1: [log_prob(steering=-0.3), log_prob(gas=0.4), log_prob(brake=0.6)]")
print("   Row 2: [log_prob(steering=0.0), log_prob(gas=0.9), log_prob(brake=0.1)]")

# Step 5: Sum across action dimensions
log_probs = log_probs_per_component.sum(dim=1)

print(f"\n4. FINAL RESULT AFTER .sum(dim=1) (shape: {log_probs.shape})")
print("   Each value = total log probability of the entire action vector")
print("   Shape: [batch_size=3]")
print(log_probs)
print("\n   Explanation:")
print("   - log_prob[0] = log_prob(steering=0.5) + log_prob(gas=0.8) + log_prob(brake=0.2)")
print("   - log_prob[1] = log_prob(steering=-0.3) + log_prob(gas=0.4) + log_prob(brake=0.6)")
print("   - log_prob[2] = log_prob(steering=0.0) + log_prob(gas=0.9) + log_prob(brake=0.1)")

# Show the actual calculation
print("\n5. VERIFICATION - Manual calculation for first action:")
print(f"   log_prob(steering=0.5) = {log_probs_per_component[0, 0]:.4f}")
print(f"   log_prob(gas=0.8)      = {log_probs_per_component[0, 1]:.4f}")
print(f"   log_prob(brake=0.2)    = {log_probs_per_component[0, 2]:.4f}")
print(f"   Sum                     = {log_probs_per_component[0].sum():.4f}")
print(f"   Final log_probs[0]     = {log_probs[0]:.4f}")

print("\n" + "=" * 60)
print("WHY SUM ACROSS DIM=1?")
print("=" * 60)
print("Since action components are independent, the joint probability is:")
print("P(action) = P(steering) * P(gas) * P(brake)")
print("Taking log: log P(action) = log P(steering) + log P(gas) + log P(brake)")
print("This is exactly what .sum(dim=1) does!")

print("\n" + "=" * 60)
print("USAGE IN REINFORCE:")
print("=" * 60)
print("These log_probs are used in the policy gradient:")
print("∇θ J(θ) = E[∇θ log πθ(a|s) * G_t]")
print("where G_t is the return (cumulative reward)")


