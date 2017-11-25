#
import numpy as np
import tensorflow as tf
import pandas as pd
import gym

from time import time

import matplotlib.pyplot as plt

def BasicScenario():

    print("CartPole main start..")
    env = gym.make('CartPole-v0')

    env.reset()
    rewards = []
    rand_actions = []

    for _ in range(100):
        #env.render()
        r_action = env.action_space.sample()
        state, reward, done, info = env.step(r_action) # take a random action

        rewards.append(reward)
        rand_actions.append(r_action)
        if done:
            print("done......")
            print("reward %d state %s" % (reward, state))
            rewards = []
            env.reset()

    return np.sum(rewards)
    #plt.plot(range(len(rewards)), np.cumsum(rewards), c="r")
    #plt.show()


def main():

    rewards_hist = []
    for i in range(50): # number of episodes
        reward_sum = BasicScenario()
        rewards_hist.append(reward_sum)

    print("avg. rewards from random action ...", np.mean(rewards_hist))

    plt.plot(range(len( rewards_hist  )), rewards_hist )
    plt.show()


if __name__ == "__main__":
    main()
