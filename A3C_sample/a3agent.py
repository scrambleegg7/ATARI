#
import gym
import signal
import threading
import scipy.signal
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

import tensorflow as tf
import numpy as np

from custom_env import CustomGym
#from common import *
"""
tf.app.flags.DEFINE_string("game", "Breakout-v0", "gym environment name")
tf.app.flags.DEFINE_string("train_dir", "./models/experiment0/", "gym environment name")
tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
tf.app.flags.DEFINE_bool("use_lstm", False, "use LSTM layer")

tf.app.flags.DEFINE_integer("t_max", 6, "episode max time step")
tf.app.flags.DEFINE_integer("t_train", 1e9, "train max time step")
tf.app.flags.DEFINE_integer("jobs", 8, "parallel running thread number")

tf.app.flags.DEFINE_integer("frame_skip", 1, "number of frame skip")
tf.app.flags.DEFINE_integer("frame_seq", 4, "number of frame sequence")

tf.app.flags.DEFINE_string("opt", "rms", "choice in [rms, adam, sgd]")
tf.app.flags.DEFINE_float("learn_rate", 7e-4, "param of smooth")
tf.app.flags.DEFINE_integer("grad_clip", 40.0, "gradient clipping cut-off")
tf.app.flags.DEFINE_float("eps", 1e-8, "param of smooth")
tf.app.flags.DEFINE_float("entropy_beta", 1e-2, "param of policy entropy weight")
tf.app.flags.DEFINE_float("gamma", 0.95, "discounted ratio")

tf.app.flags.DEFINE_float("train_step", 0, "train step. unchanged")

flags = tf.app.flags.FLAGS
"""


class A3CNet(object):

    def __init__(self, session, action_size, model='mnih',
        optimizer=tf.train.AdamOptimizer(1e-4)):

        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session

        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            #if model == 'mnih':
            self.state, self.policy, self.value = self.build_model(84, 84, 4)
            #else:
                # Assume we wanted a feedforward neural network
                #self.state, self.policy, self.value = self.build_model_feedforward(4)
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            scope='network')
            self.advantages = tf.placeholder('float32', [None], name='advantages')

        with tf.variable_scope('optimizer'):
            # Compute the one hot vectors for each action given.
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

            min_policy = 1e-8
            max_policy = 1.0 - 1e-8
            #self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))

            eps = 1e-8
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + eps))
            time_diff = self.target_value - self.value

            policy_prob = tf.log(tf.reduce_sum(tf.multiply(self.policy, action_one_hot), reduction_indices=1))
            self.policy_loss = - tf.reduce_sum(policy_prob * time_diff)
            self.value_loss = tf.reduce_sum(tf.square(time_diff))

            #self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,  scope='network'))

            entropy_beta = 1e-2
            self.total_loss = self.policy_loss + self.value_loss * 0.5 + self.entropy * entropy_beta

            grads = tf.gradients(self.total_loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))

            # Create an operator to apply the gradients using the optimizer.
            # Note that apply_gradients is the second part of minimize() for the
            # optimizer, so will minimize the loss.
            self.train_op = optimizer.apply_gradients(grads_vars)



    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()

    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    def get_policy_and_value(self, state):
        policy, value = self.sess.run([self.policy, self.value], {self.state:
        state})
        return policy.flatten(), value.flatten()

    # Train the network on the given states and rewards
    def train(self, states, actions, target_values, advantages):
        # Training
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.target_value: target_values,
            self.advantages: advantages
        })

    def build_model(self, h, w, channels):
        self.layers = {}
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
        self.layers['state'] = state
        # First convolutional layer
        with tf.variable_scope('conv1'):
            conv1 = tf.contrib.layers.convolution2d(inputs=state,
            num_outputs=16, kernel_size=[8,8], stride=[4,4], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer())
            self.layers['conv1'] = conv1

        # Second convolutional layer
        with tf.variable_scope('conv2'):
            conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=32,
            kernel_size=[4,4], stride=[2,2], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer())
            self.layers['conv2'] = conv2

            h_conv2_shape = conv2.get_shape().as_list()

            print(h_conv2_shape)
            print("conv2 dimension:",h_conv2_shape[1],h_conv2_shape[2],h_conv2_shape[3])

        # Flatten the network
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(inputs=conv2)
            self.layers['flatten'] = flatten

            f_shape = flatten.get_shape().as_list()
            print(f_shape)


        # Fully connected layer with 256 hidden units
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer())
            self.layers['fc1'] = fc1

            fc_shape = fc1.get_shape().as_list()
            print(fc_shape)


        # The policy output
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(inputs=fc1,
            num_outputs=self.action_size, activation_fn=tf.nn.softmax,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None)
            self.layers['policy'] = policy

            policy_shape = policy.get_shape().as_list()
            print(policy_shape)


        # The value output
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None)
            self.layers['value'] = value

            value_shape = value.get_shape().as_list()
            print(value_shape)

        return state, policy, value

def main():

    customgym = CustomGym('SpaceInvaders-v0')
    action_size = customgym.action_size

    s = customgym.reset_1()
    print(np.min(s),np.max(s))
    print("initial shape of state ...", s.shape)

    with tf.Session() as sess:

        myagent = A3CNet(sess,action_size)
        sess.run(tf.global_variables_initializer())

        policy, value = myagent.get_policy_and_value(s)

        print(policy, np.sum(policy))
        print(value)

        done = False
        i = 0
        R = 0
        while i < 50 and not done:

            policy, value = myagent.get_policy_and_value(s)
            action_idx = np.random.choice(action_size, p=policy)

            s_, r, done, _ = customgym.step_1(action_idx)
            #print(action_idx, r)

            s = s_
            R += r
            i += 1

            if done:
                break
        print("game reset .....", i, R)

if __name__ == "__main__":
    main()
