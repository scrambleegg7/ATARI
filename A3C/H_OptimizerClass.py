#
import numpy as np


import tensorflow as tf

import gym, time, random, threading


class OptimizerClass(threading.Thread):

    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
        	brain.optimize()

    def stop(self):
        self.stop_signal = True
