
from H_AgentClass import AgentClass
from H_BrainClass import Brain
from H_Environment import EnvironmentClass
from H_OptimizerClass import OptimizerClass

import numpy as np


import tensorflow as tf

import gym, time, random, threading


RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

def main():




if __name__ == "__main__":
    main()
