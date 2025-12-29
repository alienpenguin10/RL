import numpy as np

""" A simple recording buffer to store states, actions for analysis or replay."""
class RecordingBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = {
            'states': [],
            'actions': [],
        }

    def push(self, state, action):
        if len(self.buffer['states']) >= self.capacity:
            # Remove the oldest experience to maintain capacity
            for key in self.buffer:
                self.buffer[key].pop(0)
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)

    def push_batch(self, states, actions):
        for state, action in zip(states, actions):
            self.push(state, action)

    def get_recording(self):
        return np.array(self.buffer['states']), np.array(self.buffer['actions'])

    def __len__(self):
        return len(self.buffer['states'])
    
    def clear(self):
        self.buffer = {
            'states': [],
            'actions': [],
        }