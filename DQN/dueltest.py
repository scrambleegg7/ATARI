#

# simple version to train duel Q network

import tensorflow as tf
import numpy as np
import pandas as pd

from AgentClass_v4duel import AgentClass

import tensorflow.contrib.slim as slim

tf.reset_default_graph()

ENV_NAME = 'SpaceInvaders-v0'
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME

env = gym.make(ENV_NAME)

STATE_LENGTH=4
myAgent = AgentClass(env.action_space.n,STATE_LENGTH)

#
#parameters
#
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

def get_initial_state(observation, last_observation):

    init_image = rgb2gray(observation)
    init_image = resize(init_image, (84,84),mode="constant")
    init_image = rescale_intensity(init_image,out_range=(0,255))

    #processed_observation = np.maximum(observation, last_observation)
    #processed_observation = np.uint8(resize(rgb2gray(last_observation), (84, 84)) * 255)
    state = [init_image for _ in range(4)]
    stacked_image = np.stack(state, axis=0)

    #
    #  one layer equal to one game screen image ...
    #

    #  layer 0 --> game image (no action)
    #  layer 1 --> game image (no action)
    #  layer 2 --> game image (no action)
    #
    init_image = np.transpose(stacked_image,(1,2,0))
    # should be changed W x H x C format
    return init_image

def preprocess(observation, last_observation):
    #processed_observation = np.maximum(observation, last_observation)
    obs_image = rgb2gray(observation)
    obs_image = resize(obs_image, (84,84),mode="constant")
    obs_image = rescale_intensity(obs_image,out_range=(0,255))

    return obs_image



#Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        state = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
