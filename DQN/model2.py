import tensorflow as tf
import numpy as np

NUM_STATES = 10
NUM_ACTIONS = 2
GAMMA = 0.5


def inference(x_ph, num_states, h1, h2, num_actions,stddev = 0.01):

    NUM_INPUT = num_states
    NUM_HIDDEN1 = h1
    NUM_HIDDEN2 = h2
    NUM_OUTPUT = num_actions

    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([NUM_INPUT, NUM_HIDDEN1], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_HIDDEN1], dtype=tf.float32), name='biases')
        #hidden1 = tf.nn.relu(tf.matmul(x_ph, weights) + biases)
        hidden1 = tf.sin(tf.matmul(x_ph, weights) + biases)
        #
        # relu is not good way to coverge loss function...
        #

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN1, NUM_HIDDEN2], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_HIDDEN2], dtype=tf.float32), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN2, NUM_OUTPUT], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_OUTPUT], dtype=tf.float32), name='biases')
        y = tf.matmul(hidden2, weights) + biases

    return y



def hot_one_state(index):
    array = np.zeros(NUM_STATES)
    array[index] = 1.
    return array

# we will create a set of states, the agent get a reward for getting to the 5th one(4 in zero based array).
# the agent can go forward or backward by one state with wrapping(so if you go back from the 1st state you go to the end).
states = [(x == 4) for x in range(NUM_STATES)]
# [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

session = tf.Session()
state = tf.placeholder("float", [None, NUM_STATES])
targets = tf.placeholder("float", [None, NUM_ACTIONS])

output = inference(state,NUM_STATES,500,500,NUM_ACTIONS)

#
# just one layer is enough to solve this simple states / actions.
#

#hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))
#output = tf.matmul(state, hidden_weights)

loss = tf.reduce_mean(tf.square(output - targets))
train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

session.run(tf.initialize_all_variables())

for i in range(50):
    state_batch = []
    rewards_batch = []

    # create a batch of states
    for state_index in range(NUM_STATES):
        state_batch.append(hot_one_state(state_index))

        minus_action_index = (state_index - 1) % NUM_STATES
        plus_action_index = (state_index + 1) % NUM_STATES

        minus_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(minus_action_index)]})[0]
        plus_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(plus_action_index)]})[0]



        # these action rewards are the results of the Q function for this state and the actions minus or plus
        action_rewards = [states[minus_action_index] + GAMMA * np.max(minus_action_state_reward),
                          states[plus_action_index] + GAMMA * np.max(plus_action_state_reward)]
        rewards_batch.append(action_rewards)

    #print("minus_action_state_reward",i,minus_action_state_reward)
    #print("plus_action_state_reward",i,plus_action_state_reward)

    session.run(train_operation, feed_dict={
        state: state_batch,
        targets: rewards_batch})

print([states[x] + np.max(session.run(output, feed_dict={state: [hot_one_state(x)]}))
       for x in range(NUM_STATES)])

# The final output will look something like this, the values converage about the reward state.
# [0.16162321, 0.31524473, 0.62262321, 1.2479111, 1.6226232, 1.2479111, 0.62262321, 0.31524473, 0.16162321, 0.031517841]
