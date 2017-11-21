
import gym
import numpy as np

env = gym.make('CartPole-v0')

goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 5
last_time_steps = np.zeros(num_consecutive_iterations)


for episode in range(num_episodes):
    #
    observation = env.reset()

    episode_reward = 0
    for t in range(max_number_of_steps):
        #env.render()

        action = np.random.choice([0, 1])

        observation, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            print(observation,reward,done,info)

            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1,
                last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            break

    if (last_time_steps.mean() >= goal_average_steps):
        print('Episode %d train agent successfuly!' % episode)
        break
