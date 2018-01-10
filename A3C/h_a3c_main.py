
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

    env_test = EnvironmentClass(render=True, eps_start=0., eps_end=0.)
    NUM_STATE = env_test.env.observation_space.shape[0]
    NUM_ACTIONS = env_test.env.action_space.n
    NONE_STATE = np.zeros(NUM_STATE)

    print("NUM_STATE", NUM_STATE)
    print("NUM_ACTIONS", NUM_ACTIONS)


    brain = Brain(NUM_STATE, NUM_ACTIONS)	# brain is global in A3C

    envs = [EnvironmentClass() for i in range(THREADS)]
    opts = [OptimizerClass() for i in range(OPTIMIZERS)]

    for o in opts:
    	o.start()

    for e in envs:
    	e.start()

    time.sleep(RUN_TIME)

    for e in envs:
    	e.stop()
    for e in envs:
    	e.join()

    for o in opts:
    	o.stop()
    for o in opts:
    	o.join()

    print("Training finished")
    env_test.run()

if __name__ == "__main__":
    main()
