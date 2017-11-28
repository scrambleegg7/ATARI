
#
import numpy as np
import tensorflow as tf
import pandas as pd

import gym

from MemoryClass import Memory
from StateClass import SteteClass

from env import setBreakEnv

from AgentClass import AgentClass

def keepMemory(memory_size=10000, pretrain_length=5000,render=False):

    #print("CartPole main start..")
    #env = gym.make('CartPole-v0')

    envs = setBreakEnv()
    env = envs["BreakGame"]

    # Initialize the simulation
    #observation = env.reset()
    stateCls = SteteClass(env)
    stateCls.initial_buffer()
    state = stateCls.convertAndConcatenateBuffer()
    print("initial state size ...", state.shape)
    # Take one random step to get the pole and cart moving
    #state, reward, done, _ = env.step(env.action_space.sample())

    memory = Memory(max_size=memory_size)

    # AgentClass section
    myAgent = AgentClass(2)
    # initialize Q Network



    epsilon = 1.0
    EPSILON_DECAY = 300
    FINAL_EPS = 0.1



    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        if render:
            env.render()

        # Make a random action
        action = env.action_space.sample()

        action = myAgent.get_action(state)
        break
        #myAgent.copyTargetQNetwork()
        #var = myAgent.sess.run( myAgent.copyTargetQNetworkOperation)
        #print(var[0])


        next_state, reward, done, _ = env.step(action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

    #memory.checkBuffer()

    return memory, state, env


def main():

    mem, state, env =  keepMemory(memory_size=10,pretrain_length=10)

if __name__ == "__main__":
    main()
