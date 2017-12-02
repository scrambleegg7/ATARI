
import gym
import numpy as np

from skimage.transform import resize

import matplotlib.pyplot as plt


env = gym.make('SpaceInvaders-v0')

goal_average_steps = 195
max_number_of_steps = 2000
num_consecutive_iterations = 100
num_episodes = 100
last_time_steps = []

def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_initial_state(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
    state = [processed_observation for _ in xrange(3)]
    stacked_image = np.stack(state, axis=0)

    init_image = np.transpose(stacked_image,(1,2,0))
    # should be changed W x H x C format
    return init_image

def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
    return processed_observation
    #return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


print("SpaceInvador action number..",  env.action_space.n )

total_episode_rewards = []

for episode in range(num_episodes):
    #
    observation = env.reset()

    for _ in range(np.random.randint(1,10)):
        last_observation = observation
        observation, reward, done, info = env.step(0)

    state_image = get_initial_state(observation, last_observation)
    #plt.imshow(image[1:,:,:])
    #plt.show()
    #break

    episode_reward = 0
    for t in range(max_number_of_steps):
        #env.render()

        last_observation = observation
        action = np.random.randint(0,env.action_space.n)
        observation, reward, done, info = env.step(action)

        # action =
        episode_reward += reward

        processed_image = preprocess(observation, last_observation)

        state_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
        if done:
            #print(observation,reward,done,info)

            print('%d Episode finished after %f steps / mean %f' % (episode, t + 1,  episode_reward / (t+1)  ))
            #print("episode reward : ",episode_reward)
            total_episode_rewards.append( episode_reward / (t+1) )

            break

print("avg. reward mean / episodes ", np.mean( total_episode_rewards ))
