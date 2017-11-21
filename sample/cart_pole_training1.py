
import gym
import numpy as np

env = gym.make('CartPole-v0')

goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 10000
last_time_steps = np.zeros(num_consecutive_iterations)

Q = np.zeros([4 ** 4,env.action_space.n])


def action_updater(episode,env,parameters,last_time_steps):

    episode_reward = 0
    observation = env.reset()

    for t in range(max_number_of_steps):
        #env.render()

        # random search
        # simple another way to generate 0,1 binary mode
        #action = np.random.choice([0, 1])
        action = 0 if np.matmul(parameters,observation) < 0 else 1

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            #print('%d Episode finished after %f time steps' % (episode, t + 1))
            #print(" reward", episode_reward)
            #    last_time_steps.mean()))
            #last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            break

    return episode_reward

best_reward = 0
best_parames = None

gamma = 0.01
parameters = np.random.rand(4) * 2 - 1 # generate [-1,1] x 4

for episode in range(num_episodes):
    #

    new_parameters = parameters + gamma * (np.random.rand(4) * 2 - 1 ) # generate [-1,1] x 4
    reward = action_updater(episode,env,parameters,last_time_steps)

    if reward > best_reward:
        best_reward = reward
        best_parames = parameters
        parameters = new_parameters

        if reward == 200:
            print(episode)
            break

    #if (last_time_steps.mean() >= goal_average_steps):
    #    print('Episode %d train agent successfuly!' % episode)
    #    break
