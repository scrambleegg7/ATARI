import numpy as np
import random
from time import time
import gym
from gym import wrappers

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


def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.

    p is probality
    s is state
    r is reward



    """

    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v


def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    print("prior policy..")
    print(policy)
    max_iterations = 200000
    gamma = 1.0
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    env = gym.make('FrozenLake8x8-v0')
    #env = gym.make('CartPole-v0')

    env.seed(0)

    print(env.P[0][0])

    policy = policy_iteration(env)
    print("converged policy....")
    print(policy)

    n_policies = 2000
    start_time = time()
