#
#
#

import gym
import numpy as np

envs = {}


def setEnv():

    ENV_NAME = "Breakout-v0"
    envs["BreakGame"] = gym.make(ENV_NAME)

    ENV_NAME = "SpaceInvaders-v0"
    envs["SpaceInvador"] = gym.make(ENV_NAME)

    return envs
