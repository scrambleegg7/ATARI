#
#
#

import gym
import numpy as np

def setBreakEnv():

    envs = {}
    ENV_NAME = "Breakout-v0"
    envs["BreakGame"] = gym.make(ENV_NAME)

    return envs
