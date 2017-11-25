#

import gym
import pandas as pd
import numpy as np
import logging

from gym import wrappers

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self,observation,reward,done):
        return self.action_space.sample()

def gen_random_policy():
	return (np.random.uniform(-1,1, size=2), np.random.uniform(-1,1))

def policy_to_action(env, policy, obs):
    if np.dot(policy[0], obs) + policy[1] > 0:
    	return 1
    else:
    	return 0

def run_episode(env, policy, t_max=1000, render=False):
    obs = env.reset()
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
        selected_action = policy_to_action(env, policy, obs)
        obs, reward, done, _ = env.step(selected_action)
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == '__main__':
    print("main start")
    env = gym.make('MountainCar-v0')
    #gym.undo_logger_setup()
    logger = logging.getLogger()
    #formatter = logger.formatter()
    logger.setLevel(logging.INFO)

    outdir = "output/random-agent-results"
    wrappers.Monitor(env, directory=outdir,force=True)
    env.seed(0)
    _agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False

    for i in range(episode_count):
        observation = env.reset()
        env.render()
        while True:
            action = _agent.act(observation,reward, done)
            observation, reward, done, _ = env.step(action)
            if done:
                break

    env.close()
    # Generate a pool or random policies
	#n_policy = 500
	#policy_list = [gen_random_policy() for _ in range(n_policy)]

	# Evaluate the score of each policy.
	#scores_list = [run_episode(env, p, 1000) for p in policy_list]

	# Select the best plicy.
	#print('Best policy score = %f' %max(scores_list))

	#best_policy= policy_list[np.argmax(scores_list)]
	#print('Running with best policy:\n')
    #run_episode(env, best_policy, render=True)
