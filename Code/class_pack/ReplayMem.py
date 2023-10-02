import math
import random
import numpy as np
from collections import namedtuple
Transition = namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_self','dist_weight_obs','action', 'next_state_o','next_state_o_obs','next_dist_weight','next_dist_weight_self','next_dist_weight_obs', 'reward'))
#Transition = namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_obs','agent_obs_nbr','action', 'next_state_o','next_state_o_obs','next_dist_weight','next_dist_weight_obs','next_agent_obs_nbr', 'reward'))
Transition_predicate = namedtuple('Transition',('state_o','state_o_obs','dist_weight','dist_weight_obs','action'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Predict_Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition_predicate(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def clear_memory(self):
        self.memory = []
        self.position = 0
        
    
    # def sample_test(self, batch_size):
    #     if int(len(self.memory)*0.1) > batch_size+1:
    #         return random.sample(self.memory[int(len(self.memory)*0.9):], batch_size)
    #     else:
    #         return self.memory[int(len(self.memory)*0.9):]

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    rep_test=ReplayMemory(10000)
    print(len(rep_test))
    
