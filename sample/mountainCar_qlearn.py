import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import logging

#sns.set(color_codes=True)

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s' )

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

#q_table = np.random.uniform(low=-100, high=-1, size=(num_states, num_states, env.action_space.n))
q_table = np.zeros( (num_states,num_states,env.action_space.n)  )

def run_episode(policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    #gamma = 1.
    for _ in range(max_number_of_steps):
        if render:
            env.render()
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

    print("total steps:%d  total reward:%d" % (step_idx, total_reward) )
    return total_reward

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):

    obs = observation

    obs0 = int( np.digitize(obs[0], bins=bins(env_low[0], env_high[0], num_states)) )
    obs1 = int( np.digitize(obs[1], bins=bins(env_low[1], env_high[1], num_states) ) )

    return obs0, obs1

def get_random_action():

    return np.random.choice(range(env.action_space.n))

def get_action(obs0,obs1,i):


    #
    # greedy aproach
    #
    r = np.random.uniform(0,1)
    epsilon = 0.5 * (0.9 ** i)
    if epsilon <= r:
        # softmax ....
        actions = np.exp( q_table[ obs0, obs1, : ] )
        probs = actions / np.sum(actions)
        next_action = np.random.choice(range(env.action_space.n), p= probs)
    else:
        next_action = get_random_action()

    return next_action

logging.debug("number of actions : %s", env.action_space.n)
logging.debug("Q shape : %s", q_table.shape)

alpha = 1.
gamma = 1.

for episode in range(num_episodes):
    # initialize state
    observation = env.reset()
    #action = np.argmax(q_table[state])
    episode_rewards = []
    positions = []
    episode_reward = 0
    for t in range(max_number_of_steps):
        #env.render()
        obs0,obs1 = digitize_state(observation)
        action = get_action(obs0,obs1,episode)

        #logging.debug("action %s", action)
        observation, reward, done, info = env.step(action)
        next_obs0, next_obs1 = digitize_state(observation)
        # get observation , reward
        #positions.append(pos)
        #eta = max(0.003, alpha * (0.99 ** (episode//100)))
        eta = alpha * (0.99 ** (episode//100)))

        q_table[obs0, obs1, action] = (1 - eta) * q_table[ obs0, obs1,  action] +\
                eta * (reward + gamma * np.max( q_table[next_obs0, next_obs1, :] ) )

        episode_rewards.append(reward)

        if done:
            break

    if episode % 100 == 0:
        print("Done.. episode:%d steps:%d reward:%d" % (episode, t+1, np.sum(episode_rewards)) )

print("trainig done ....")
new_policy = np.argmax(q_table,axis=2)
#print("new_policy ...", new_policy)

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
cm = plt.get_cmap("Paired")
sns.heatmap(new_policy, square=True, cmap="Reds", center=1)
plt.show()




scenario_rewards = [ run_episode(new_policy) for _ in range(100)      ]
print("** Rewards after training...", np.mean( scenario_rewards) )

np.save("mountaincar_policy.npy",new_policy)
