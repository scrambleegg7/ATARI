
#    Different actions yield different rewards.
#    For example, when looking for treasure in a maze,
#    going left may lead to the treasure,
#    whereas going right may lead to a pit of snakes.
#    Rewards are delayed over time.
#    This just means that even if going left in the above example is
#    the right things to do, we may not know it till later in the maze.
#    Reward for an action is conditional on the state of the environment.
#    Continuing the maze example,
#    going left may be ideal at a certain fork in the path, but not at others.

#     Loss = -log(PAI)*A

#    A is advantage,
#    and is an essential aspect of all reinforcement learning algorithms.
#    Intuitively it corresponds to how much better an action was than some baseline.
#    In future algorithms,
#    we will develop more complex baselines to compare our rewards to,
#    but for now we will assume that the baseline is 0,
#    and it can be thought of as simply the reward we received for each action.


# this is simulation 4 slotmachine generating different probabilities....


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


#List out our bandit arms.
#Currently arm 4 (index #3) is set to most often provide a positive reward.
bandit_arms = [0.2,0,-0.2,-2]
num_arms = len(bandit_arms)
def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1



#The Agent

#The code below established our simple neural agent.
#It consists of a set of values for each of the bandit arms.
#Each value is an estimate of the value of the return from choosing the bandit.
#We use a policy gradient method to update the agent
#by moving the value for the selected action toward the recieved reward.
tf.reset_default_graph()

#These two lines established the feed-forward part of the network.
weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

#The next six lines establish the training proceedure.
#We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network.

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_output = tf.slice(output,action_holder,[1])
loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)

#Training the Agent

#We will train our agent by taking actions in our environment,
#and recieving rewards.
#Using the rewards and actions,
#we can know how to properly update our network
#in order to more often choose actions that will yield the highest rewards over time.

total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_arms) #Set scoreboard for bandit arms to 0.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:

        #Choose action according to Boltzmann distribution.
        actions = sess.run(output)
        a = np.random.choice(actions,p=actions)
        action = np.argmax(actions == a)

        reward = pullBandit(bandit_arms[action]) #Get our reward from picking one of the bandit arms.

        #Update the network.
        _,resp,ww = sess.run([update,responsible_output,weights], feed_dict={reward_holder:[reward],action_holder:[action]})

        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
        i+=1
print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
