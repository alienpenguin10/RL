import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = int(capacity)
        self.device = device
        
        # Current size of the buffer (number of transitions stored)
        self.size = 0
        # Position of the next transition to be stored (circular index)
        self.ptr = 0

        # --- Pre-allocate NumPy arrays for high-speed storage ---
        # State: Use the stacked frame shape (e.g., 4 x 96 x 96)
        self.state = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        
        # Next State: Also uses the stacked frame shape
        self.next_state = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        
        # Action: Continuous 3D vector (steering, gas, brake)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        
        # Reward: Scalar
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        
        # Done: Boolean flag indicating episode termination
        self.done = np.zeros(self.capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """
        Stores a single transition tuple (s, a, r, s', done) into the buffer.
        """
        # Store data at the current pointer index
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        # Convert boolean 'done' to float (0.0 or 1.0)
        self.done[self.ptr] = float(done)

        # Move the pointer to the next position (circularly)
        self.ptr = (self.ptr + 1) % self.capacity
        # Increment the size up to the max capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions for training.
        """
        # Select random indices from the currently filled portion of the buffer
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Convert NumPy data to PyTorch Tensors on the correct device (GPU)
        state = torch.FloatTensor(self.state[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        reward = torch.FloatTensor(self.reward[ind]).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(self.next_state[ind]).to(self.device)
        done = torch.FloatTensor(self.done[ind]).unsqueeze(1).to(self.device)
        
        # The unsqueeze(1) makes the reward and done tensors have shape [batch_size, 1] 
        # instead of [batch_size], which is necessary for correct vector arithmetic in the loss functions.

        return (state, action, reward, next_state, done)

    def __len__(self):
        return self.size