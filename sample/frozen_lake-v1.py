import numpy as np
import time

import gym

def run_episode(env, policy, episode_len=100, render=False):
    total_reward = 0
    obs = env.reset()
    for t in range(episode_len):
        if render:
            env.render()
        action = policy[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes

def gen_random_policy():
    return np.random.choice(4, size=((16)))

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    ## Policy search
    n_policies = 2000
    start = time.time()
    policy_set = [gen_random_policy() for _ in range(n_policies)]
    policy_score = [evaluate_policy(env, p) for p in policy_set]
    end = time.time()

print("Best score = %0.2f. Time taken = %4.4f seconds" %(np.max(policy_score) , end - start))
