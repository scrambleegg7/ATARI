#

import gym
import numpy as np

from skimage.transform import resize

import matplotlib.pyplot as plt

import tensorflow as tf

import gym_pull
#gym_pull.pull('github.com/ppaquette/gym-super-mario')
# Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('MsPacman-v0')

goal_average_steps = 195
max_number_of_steps = 2000
num_consecutive_iterations = 100
num_episodes = 10
last_time_steps = []

def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_initial_state(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
    state = [processed_observation for _ in range(3)]
    stacked_image = np.stack(state, axis=0)

    init_image = np.transpose(stacked_image,(1,2,0))
    # should be changed W x H x C format
    return init_image

def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
    return processed_observation
    #return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


print("MarioBrother action number..",  env.action_space.n )

total_episode_rewards = []

episode_reward_mean_ph = tf.placeholder(tf.float32)
episode_reward_mean = tf.Variable(.0)
tf.summary.scalar('episode_reward_mean', episode_reward_mean)

update_episode_reward_ops = episode_reward_mean.assign(episode_reward_mean_ph)

with tf.Session() as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tfmodel/test',
                                      sess.graph)

    for episode in range(num_episodes):
        #
        observation = env.reset()
        #print("observation shape" , observation.shape)
        #plt.imshow(observation)
        #plt.show()
        #break

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

            #if reward > 0:
                #print("reward is not ZERO", t, reward)

            processed_image = preprocess(observation, last_observation)

            state_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
            if done:
                #print(observation,reward,done,info)
                reward_mean = episode_reward / (t+1)
                print('%d Episode finished after %f steps / mean %f' % (episode, t + 1,  reward_mean  ))
                print("   total reward : ", )
                total_episode_rewards.append( reward_mean )

                #my_feed_dict = {episode_reward_mean_ph:reward_mean}
                #sess.run( update_episode_reward_ops, feed_dict=my_feed_dict )

                #summary = sess.run(merged)
                #train_writer.add_summary(summary, 10)

                break

    print("avg. reward mean / episodes ", np.mean( total_episode_rewards ))
