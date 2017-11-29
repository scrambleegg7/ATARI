#
import tensorflow as tf
import pandas as pd
import numpy as np



# Reinforcement learning algorithms can have stability issues
# due to correlations between states

# Memory will store our experiences, our transitions s, a, r, s'

from collections import deque
class Memory():
    
    def __init__(self, max_size = 1000):
        #with deque, buffer is always updated with new one....
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        # if data full in deque tube, old data is pushed out from it.
        # then insert new item into tube...
        self.buffer.append(experience)

    def checklength(self):

        return len(self.buffer)

    def checkBuffer(self):

        for idx, i in enumerate( list(self.buffer) ):
            print("buf indx:%d" % idx)
            print(i)

    def sample(self, batch_size):
        #
        # pick up random data from tube with batch_size
        #
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]
