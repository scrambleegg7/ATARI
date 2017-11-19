#

import gym
import matplotlib.pyplot as plt



env = gym.make('FrozenLake-v0')

observation = env.reset()

for t in range(1000):

    env.render()
    _screen, reward, terminal, step_info = env.step(env.action_space.sample()) # take a random action
    if reward > 0:
        print(reward)
        print(observation.shape)
        plt.imshow(observation)
        plt.show()

    if terminal:
        print("Episode finished after {} timesteps".format(t+1))
        break
