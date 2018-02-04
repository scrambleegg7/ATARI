
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


class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)

        name = "fully_connected_model"
        with tf.variable_scope(name) as scope:

            name = "layer1"
            with tf.variable_scope(name) as scope:

                w1_shape = [s_size,h_size]
                W1 = weight_variable(shape=w1_shape)
                b1 = bias_variable([h_size])
                hidden = tf.nn.relu(  tf.matmul(self.state_in,W1) + b1 )

                w1_smry = tf.summary.histogram("W1", W1)

            name = "layer2"
            with tf.variable_scope(name) as scope:

                w2_shape = [h_size,a_size]
                W2 = weight_variable(shape=w2_shape)
                b2 = bias_variable(shape=[a_size])

                w2_smry = tf.summary.histogram("W2", W2)

        self.output = tf.nn.softmax(  tf.matmul(hidden,W2) + b2 )



        #hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        #self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)

        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

class contextual_bandit():

    def __init__(self):
        self.state = 0
        #List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
        self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        self.state = np.random.randint(0,len(self.bandits)) #Returns a random state for each episode.
        # this example gives 3 bandits then return random int number 0 or 1 or 2...

        return self.state

    def pullArm(self,action):

        # input : action (int)
        #Get a random number.

        bandit = self.bandits[self.state,action]
        result = np.random.randn(1)  # generate n ~ [-infinite, infinite]
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1

#The Agent


#The Policy-Based Agent

# The code below established our simple neural agent.
# It takes as input the current state, and returns an action.
# This allows the agent to take actions which are conditioned on the state of the environment,
# a critical step toward being able to solve full RL problems.
# The agent uses a single set of weights,
# within which each value is an estimate of the value of the return from choosing a particular arm given a bandit.
# We use a policy gradient method to update the agent by moving the value for the selected action toward the recieved reward.

#env = gym.make('CartPole-v0')

tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_lenght.append(j)
                break


            #Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
