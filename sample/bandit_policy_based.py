
# Here we define our contextual bandits. In this example,
# we are using three four-armed bandit.
# What this means is that each bandit has four arms that can be pulled.
# Each bandit has different success probabilities for each arm,
# and as such requires different actions to obtain the best result.
# The pullBandit function generates a random number from a normal distribution with a mean of 0.
# The lower the bandit number, the more likely a positive reward will be returned.
# We want our agent to learn to always choose the bandit-arm that will most often give a positive reward,
# depending on the Bandit presented.

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


def act():

    action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

    min_policy = 1e-8
    max_policy = 1.0 - 1e-8
    self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))

    # For a given state and action, compute the log of the policy at
    # that action for that state. This also works on batches.
    self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices=1)



gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def weight_variable(shape):
    W = tf.get_variable("W", shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
    return W

def bias_variable(shape):
    initial_value = tf.truncated_normal(shape, 0.0, 0.001)
    b = tf.get_variable("b",initializer=initial_value)
    return b


class BanditClass(object):

    def __init__(self,num_bandits, num_actions):

        self.num_bandits = num_bandits
        self.num_actions = num_actions

        # Placeholders
        self.state = tf.placeholder(name='state',
            shape=[1], dtype=tf.int32)
        self.action = tf.placeholder(name='action',
            shape=[1], dtype=tf.int32)
        self.reward = tf.placeholder(name='reward',
            shape=[1], dtype=tf.float32)

        # One hot encode the state
        self.state_one_hot = tf.one_hot(indices=self.state, depth=num_bandits)

        # Feed forward net to choose the action
        with tf.variable_scope("net"):
            self.W_input = tf.Variable(tf.ones([num_bandits, num_actions]))
            z1 = tf.matmul(self.state_one_hot, self.W_input)
            #self.fc1 = tf.nn.sigmoid(z1)
            self.fc1 = tf.nn.softmax(z1)

        self.chosen_weight = tf.slice(tf.reshape(self.fc1, [-1, ]),
                self.action, [1])

        self.loss = -(tf.log(self.chosen_weight) * self.reward)
        self.train_optimizer = tf.train.GradientDescentOptimizer(
                                0.001).minimize(self.loss)


    def getBanditState(self):


        self.state = np.random.randint(0,len(self.bandits)) #Returns a random state for each episode.
        # this example gives 3 bandits then return random int number 0 or 1 or 2...

        return self.state

    def pullArm(self, bandit ,action):

        # input : action (int)
        #Get a random number.

        #bandit = self.bandits[self.state,action]
        result = np.random.randn(1)  # generate n ~ [-infinite, infinite]
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1

    def step1(self, sess, state, action, reward):
        pass

    def actions(self, sess, state):

        feeddict = {}
        if np.random.rand(1) < 0.25:
            action = np.random.randint(0, self.num_actions)
        else:
            #logging.debug("action from fc1 : 75%")
            feeddict[self.state] = state
            action = np.argmax(sess.run(self.fc1, feed_dict= feeddict))

        return action


    def step(self, sess, state, action, reward):


        input_feed = {self.state: state,
            self.action: action, self.reward: reward}
        output_feed = [self.W_input, self.train_optimizer]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0], outputs[1]

#The Agent


#The Policy-Based Agent

# The code below established our simple neural agent.
# It takes as input the current state, and returns an action.
# This allows the agent to take actions which are conditioned on the state of the environment,
# a critical step toward being able to solve full RL problems.
# The agent uses a single set of weights,
# within which each value is an estimate of the value of the return from choosing a particular arm given a bandit.
# We use a policy gradient method to update the agent by moving the value for the selected action toward the recieved reward.


def main():

    logging.debug("start....")

    state = 0
    #List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
    bandits = np.array([[-5, -1, 0, 1],[-1, -5, 1, 0],[0, 1, -1, -5]])

    #self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
    num_bandits = bandits.shape[0]
    num_actions = bandits.shape[1]

    logging.debug("num of bandits %d", num_bandits)
    logging.debug("num of actions %d", num_actions)

    rewards = np.zeros([num_bandits, num_actions])

    env = BanditClass(num_bandits, num_actions)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        logging.debug("tensor session start ...")
        sess.run(init)

        for i in range(10000):

            # Pick a random state [0 or 1 or 2]
            # state matrix prepared is 3 states  x 4 actions
            state = np.random.randint(0, num_bandits)
            action = env.actions(sess, [state])

            bandit = bandits[state,action]
            reward = env.pullArm(bandit, action)
            #logging.debug("state:%s action: %s reward:%s", state, action, reward)

            rewards[state,action] += reward

            w_, _ = env.step(sess,[state],[action],[reward])
            #s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.

        logging.debug("W:\n%s", w_)
        logging.debug("final rewards:\n%s", rewards)





if __name__ == "__main__":
    main()
