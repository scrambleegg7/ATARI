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

import matplotlib.pyplot as plt

#import tensorflow.contrib.slim as slim

tf.reset_default_graph()

ENV_NAME = "MsPacman-v0"
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
annealing_steps = 1000. #How many steps of training to reduce startE to endE.
num_episodes = 1000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network




q_predict,y_q_values,var_q, var_q_name = myAgent.build_Q(STATE_LENGTH)

target_predict,y_q_target,var_q_targets,var_targets = myAgent.build_target(STATE_LENGTH)

tau = .001
# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(var_q_name[var_name] * tau + (1 - tau) * target_var )
            for var_name, target_var in var_targets.items()]


#for var_name, target_var in var_targets.items():
#    print(var_name,target_var)


#print(copy_ops)

copy_online_to_target = tf.group(*copy_ops)

# Now for the training operations
learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(y_q_values * tf.one_hot(X_action, env.action_space.n),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
memory = Memory(max_size  = (pre_train_steps * 2)

mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

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
    #return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
               tf.local_variables_initializer())
sess.run(init_op)

num_episodes = 50
memory_sptes = 0

eps_min = 0.1
eps_max = 1.0



for e in range( num_episodes ):

    observation = env.reset()
    for _ in range(np.random.randint(1,10)):
        last_observation = observation
        observation, reward, done, info = env.step(0)

    print("** initial state image ...")
    state_image = get_initial_state(observation, last_observation)

    loop_steps = 0
    episode_reward = 0
    while not done:

        if memory_sptes < pre_train_steps:
            epsilon = max(eps_min, myAgent.epsilon)
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n) # random action
        else:
            action, q_values = sess.run([q_predict,y_q_values],feed_dict={myAgent.x: [state_image]})

        memory_sptes += 1
        loop_steps += 1

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        processed_image = preprocess(observation, last_observation)

        if done:
            print("episode : %d" % (e + 1))
            print("average reward %.5f" % (episode_reward / loop_steps)   )
            print("memory steps : %d" % memory_sptes)
            next_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
            state_image /= 255.0
            next_image /= 255.0
            memory.add((state_image, action, reward, done, next_image))
