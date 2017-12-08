#

# simple version to train duel Q network
import gym

import tensorflow as tf
import numpy as np
import pandas as pd

from MemoryClass import Memory
from AgentClass_v4duel import AgentClass

import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity


#import tensorflow.contrib.slim as slim

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_initial_state(observation):

    init_image = rgb2gray(observation)
    init_image = resize(init_image, (84,84),mode="constant")
    init_image = rescale_intensity(init_image,out_range=(0,255))

    #processed_observation = np.maximum(observation, last_observation)
    #processed_observation = np.uint8(resize(rgb2gray(last_observation), (84, 84)) * 255)
    state = [init_image for _ in range(STATE_LENGTH)]
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

def preprocess(observation):
    #processed_observation = np.maximum(observation, last_observation)
    obs_image = rgb2gray(observation)
    obs_image = resize(obs_image, (84,84),mode="constant")
    obs_image = rescale_intensity(obs_image,out_range=(0,255))

    return obs_image

#startE = 1.0
#endE = 0.1
#annealing_steps = 10000
#Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0
MINIBATCH_SIZE = 32

memory_size = 30000
memory = Memory(max_size=memory_size)



for i in range(num_episodes):
        myMemory = Memory()
        #Reset environment and get first new observation
        d = False
        rAll = 0
        j = 0
        episode_reward = 0

        observation = env.reset()
        #img = observation[1:176:2,::2]
        #print(img.shape)
        #plt.imshow(img)
        #plt.show()
        #break
        for _ in range(np.random.randint(1,10)):
            last_observation = observation
            observation, reward, done, info = env.step(0)

            #print( np.min(observation.ravel()), np.max(observation.ravel()) )

        #print("** making initial state image...")
        state_image = get_initial_state(observation)
        state_image /= 255.0

        while not done: # j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network

            if np.random.rand(1) < e or total_steps < pre_train_steps:
                action = np.random.randint(0,4)
            else:
                state_image = state_image[np.newaxis,:,:,:]
                qout = myAgent.sess.run(myAgent.y_q_values, feed_dict={myAgent.x:state_image}  )
                action = np.argmax(qout)
                #print("** selected action : ", a)
                #print(qout)
                #a = myAgent.sess.run(myAgent.predict,feed_dict={myAgent.x:state_image})[0]

            observation, reward, done, info = env.step(action)
            # total step + 1
            total_steps += 1

            # episode reward +
            episode_reward += reward
            processed_image = preprocess(observation)

            next_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
            state_image /= 255.0
            next_image /= 255.0
            memory.add((state_image, action, reward, done, next_image))

            # pre_train_steps = 10000
            if total_steps > pre_train_steps:

                # initial : e & endE
                # e = 1.0
                # endE = 0.1
                if e > endE:
                    e -= stepDrop

                #
                # every total_steps by 4, batch is run
                #
                if total_steps % (update_freq) == 0:

                    mini_batch = memory.sample(batch_size=MINIBATCH_SIZE)

                    states = np.array([each[0] for each in mini_batch])
                    actions = np.array([each[1] for each in mini_batch])
                    rewards = np.array([each[2] for each in mini_batch])
                    dones = np.array([each[3] for each in mini_batch])
                    next_states = np.array([each[4] for each in mini_batch])

                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = myAgent.sess.run(myAgent.q_predict,feed_dict={myAgent.x:states})
                    Q2 = myAgent.sess.run(myAgent.y_target,feed_dict={myAgent.x:next_states})
                    end_multiplier = -(dones - 1)
                    doubleQ = Q2[range(MINIBATCH_SIZE),Q1]

                    # y = Discount factor 0.99
                    targetQ = rewards + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    loss, _ = myAgent.sess.run([myAgent.loss, myAgent.grad_update], \
                        feed_dict={myAgent.a_place:actions,
                                   myAgent.y_place:targetQ,
                                   myAgent.x:states})

                    print("loss ....",loss)

                if total_steps % 10000 == 0:
                    myAgent.copyTargetQNetwork() #Update the target network toward the primary network.

            if done:
                print("total_steps....", total_steps)
