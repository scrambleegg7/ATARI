
#
import numpy as np
import tensorflow as tf
import pandas as pd

import gym

from MemoryClass import Memory
from StateClass import SteteClass

from env import setEnv

from AgentClass import AgentClass

def keepMemory(memory_size=10000, pretrain_length=5000,render=False):

    #print("CartPole main start..")
    #env = gym.make('CartPole-v0')

    envs = setEnv()

    #env = envs["BreakGame"]
    env = envs["SpaceInvador"]

    # Initialize the simulation
    #observation = env.reset()
    stateCls = SteteClass(env)
    stateCls.initial_buffer()
    curr_state = stateCls.convertAndConcatenateBuffer()

    #print("initial state size ...", state.shape)
    # Take one random step to get the pole and cart moving
    #state, reward, done, _ = env.step(env.action_space.sample())

    memory = Memory(max_size=memory_size)

    # AgentClass section
    myAgent = AgentClass(6)
    # initialize Q Network



    epsilon = 1.0
    EPSILON_DECAY = 300
    FINAL_EPS = 0.1

    NUM_FRAMES = 3

    alive_frame = 0
    total_reward = 0
    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        #if render:
        #    env.render()
        #stateCls.render()

        state = stateCls.convertAndConcatenateBuffer()
        action, q_values = myAgent.get_action(state)

        #print("** action and q_value ... ",action, q_values)
        #myAgent.copyTargetQNetwork()
        #return False,False,False
        #next_state, reward, done, _ = env.step(action)

        obs,rewards,done = stateCls.add_frame(action,NUM_FRAMES)
        print("** rewards from 3 frames ..", rewards)


        if done:
            # The simulation fails so no next state
            stateCls.envReset()
            # Start new episode
            # Take one random step to get the pole and cart moving
            alive_frame = 0
            total_reward = 0



    #memory.checkBuffer()

    return memory, state, env


def main():

    mem, state, env =  keepMemory(memory_size=10,pretrain_length=100)

if __name__ == "__main__":
    main()
