# ReplayMemory.py - DDPG experience replay buffer
import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialize experience replay buffer
        :param capacity: maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)  # double-ended queue, automatically maintains max length
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer
        :param state: current state
        :param action: executed action
        :param reward: obtained reward
        :param next_state: next state
        :param done: whether terminated (terminal state)
        """
        # convert data to numpy arrays (adapt to common reinforcement learning format)
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array(done, dtype=np.bool_)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences
        :param batch_size: batch size
        :return: batch of states/actions/rewards/next_states/dones
        """
        batch = random.sample(self.buffer, batch_size)
        # unpack and stack into batch data
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.stack(actions),
            np.stack(rewards),
            np.stack(next_states),
            np.stack(dones)
        )
    
    def __len__(self):
        """Return current number of experiences in the buffer"""
        return len(self.buffer)

# Test ReplayMemory
if __name__ == "__main__":
    memory = ReplayMemory(capacity=1000)
    # add test experiences
    memory.add([1,2,3], [0.1,0.2], 1.0, [4,5,6], False)
    memory.add([2,3,4], [0.2,0.3], 0.5, [5,6,7], True)
    # sample test
    batch = memory.sample(batch_size=2)
    print("Sampled states batch:", batch[0])
    print("Current buffer length:", len(memory))