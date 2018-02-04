import gym
import numpy as np
#import seaborn as sns

import logging

import matplotlib.pyplot as plt
import matplotlib.animation as animation


logger = logging.getLogger()
#formatter = logger.formatter()
logger.setLevel(logging.INFO)

env = gym.make('MountainCar-v0')
#env.env

env_low = env.observation_space.low
env_high = env.observation_space.high

logging.debug("env low : \n%s",env_low)
logging.debug("env high : \n%s",env_high)

goal_average_steps = 195
max_number_of_steps = 1000
num_consecutive_iterations = 1000
num_episodes = 5000
last_time_steps = np.zeros(num_consecutive_iterations)

num_states = 50

frames = []

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):

    obs = observation

    obs0 = int( np.digitize(obs[0], bins=bins(env_low[0], env_high[0], num_states)) )
    obs1 = int( np.digitize(obs[1], bins=bins(env_low[1], env_high[1], num_states) ) )

    return obs0, obs1


def run_episode(policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    #gamma = 1.
    for _ in range(max_number_of_steps):
        if render:
            #env.render()
            frames.append(env.render(mode = 'rgb_array'))
        if policy is None:
            action = env.action_space.sample()
        else:
            ob0,ob1 = digitize_state(obs)
            action = policy[ob0,ob1]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_idx += 1
        if done:
            break

    env.render(close=True)

    fig = plt.gcf()
    patch = plt.imshow(frames[0])
    plt.axis('off')


    def animate(i):
        patch.set_data(frames[i])



    #ani = animation.ArtistAnimation(fig, frames)
    ani = animation.FuncAnimation(fig, animate, frames = len(frames), interval = 50)
    #ani.save('mountaincar_anime.mp4', writer="ffmpeg")
    ani.save('mountaincar_anime.gif', writer="imagemagick")


    print("total steps:%d  total reward:%d" % (step_idx, total_reward) )
    return total_reward




policy = np.load("mountaincar_policy.npy")

#disp_heatmap(policy)


run_episode(policy,render=True)
