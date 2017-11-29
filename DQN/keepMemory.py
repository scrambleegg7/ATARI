
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

    # current state == initial screen state --> nothing to active 0 action
    curr_state = stateCls.convertAndConcatenateBuffer()
    curr_state = curr_state[np.newaxis,:,:,:]

    #print("initial state size ...", state.shape)
    # Take one random step to get the pole and cart moving
    #state, reward, done, _ = env.step(env.action_space.sample())

    memory = Memory(max_size=memory_size)

    # AgentClass section
    myAgent = AgentClass(6)
    # initialize Q Network

    MINIBATCH_SIZE = 32
    MIN_OBSERVATION = 500

    epsilon = 1.0
    EPSILON_DECAY = 300
    FINAL_EPS = 0.1

    NUM_FRAMES = 3

    observation_num = 0
    alive_frame = 0
    total_reward = 0

    curr_state_actions = []

    MEMORY_FULL = False
    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        #if render:
        #    env.render()
        #stateCls.render()

        init_state = stateCls.convertAndConcatenateBuffer()
        action, q_values = myAgent.get_action(curr_state)
        curr_state_actions.append(action)

        #print("** action and q_value ... ",action, q_values)
        #myAgent.copyTargetQNetwork()
        #return False,False,False
        #next_state, reward, done, _ = env.step(action)

        obs,rewards,done = stateCls.add_frame(action,NUM_FRAMES)

        #if observation_num % 500 == 0:
        #    print("observation_num / q_values ..",observation_num,q_values)

        if done:
            # The simulation fails so no next state
            if MEMORY_FULL:
                print("memory full.....")

            print("** rewards from done ...", total_reward)
            print("** maxium lived frame .. ", alive_frame)


            stateCls.envReset()
            # Start new episode
            # Take one random step to get the pole and cart moving
            alive_frame = 0
            total_reward = 0

        new_state = stateCls.convertAndConcatenateBuffer()
        #memory add
        memory.add((init_state, action, rewards, done, new_state))
        total_reward += rewards

        if memory.checklength() > MIN_OBSERVATION:
            MEMORY_FULL = True
            # Sample mini-batch from memory
            mini_batch = memory.sample(MINIBATCH_SIZE)
            myAgent.train(mini_batch)

            #s_batch, a_batch, r_batch, d_batch, s2_batch = memory.sample(MINIBATCH_SIZE)
            #self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
            #self.deep_q.target_train()


        observation_num += 1
        alive_frame += 1


    print(memory.checklength())
    #print("curr action", curr_state_actions)

    #print("Total rewards from all episodes..", total_reward)

    return curr_state_actions


def main():

    actions =  keepMemory(memory_size=1000,pretrain_length=1000)

if __name__ == "__main__":
    main()
