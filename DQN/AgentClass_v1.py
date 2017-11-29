#
import numpy as np
import tensorflow as tf
import pandas as pd


from collections import deque

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1  #
EXPLORATION_STEPS = 1000000  #

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_LENGTH  = 3

class AgentClass(object):

    def __init__(self, num_actions):

        self.num_actions = num_actions  #
        self.epsilon = INITIAL_EPSILON  # g
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS  #
        self.time_step = 0
        self.repeated_action = 0

        self.x = tf.placeholder( "float", [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH ] )
        print("** AgentClass x: ",self.x.get_shape().as_list())

        y_conv, var = self.build_network()


    def epsilon_update(self):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_step
        return self.epsilon

    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)

    def test(self):
        with tf.variable_scope("test") as scope:
            name = "conv1" + main_name
            with tf.variable_scope(name) as scope:

                with tf.variable_scope("w") as scope:
                    initial_value = tf.truncated_normal(shape, stddev=0.1)
                    W = tf.get_variable("W",initializer=initial_value)



    def build_network(self,main_name="Q"):

        # reuse is used to share weight and other values..

        with tf.variable_scope(main_name) as main_scope:

            x_image = tf.reshape(self.x, [-1, 84, 84, STATE_LENGTH])

            name = "conv1" + main_name
            with tf.variable_scope(name) as conv1_scope:
                W_conv1 = self.weight_variable([8, 8, STATE_LENGTH, 32])
                b_conv1 = self.bias_variable([32])
                #conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1,[1,4,4,1] ) + b_conv1)
                conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1,[1,4,4,1] ) + b_conv1)

            name = "conv2" + main_name
            with tf.variable_scope(name) as conv2_scope:
                W_conv2 = self.weight_variable([4, 4, 32, 64])
                b_conv2 = self.bias_variable([64])
                conv2 = tf.nn.relu(self.conv2d(conv1, W_conv2,[1,2,2,1]) + b_conv2)

            name = "conv3" + main_name
            with tf.variable_scope(name) as conv3_scope:
                W_conv3 = self.weight_variable([3, 3, 64, 64])
                b_conv3 = self.bias_variable([64])
                conv3 = tf.nn.relu(self.conv2d(conv2, W_conv3,[1,1,1,1]) + b_conv3)

            h_conv3_shape = conv3.get_shape().as_list()
            #print(h_conv3_shape)
            #print("dimension:",h_conv3_shape[1],h_conv3_shape[2],h_conv3_shape[3])

            name = "fc1" + main_name
            with tf.variable_scope(name) as fc1_scope:
                W_fc1 = self.weight_variable([7 * 7 * 64, 512])
                b_fc1 = self.bias_variable([512])
                h_flat = tf.reshape(conv3, [-1, 7*7*64])
                h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

            name = "fc2" + main_name
            with tf.variable_scope(name) as fc2_scope:
                W_fc2 = self.weight_variable([512, self.num_actions])
                b_fc2 = self.bias_variable([self.num_actions])

                y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,main_name)
        # return fc final layer and var
        return y_conv, var

    def build_target(self,main_name="target"):

        #  is used to share weight and other values..

        with tf.variable_scope(main_name) as scope:

            x_image = tf.reshape(self.x, [-1, 84, 84, STATE_LENGTH])

            name = "conv1" + main_name
            with tf.variable_scope(name) as scope:
                W_conv1 = self.weight_variable([8, 8, STATE_LENGTH, 32])
                b_conv1 = self.bias_variable([32])
                #conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1,[1,4,4,1] ) + b_conv1)
                conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1,[1,4,4,1] ) + b_conv1)

            name = "conv2" + main_name
            with tf.variable_scope(name) as scope:
                W_conv2 = self.weight_variable([4, 4, 32, 64])
                b_conv2 = self.bias_variable([64])
                conv2 = tf.nn.relu(self.conv2d(conv1, W_conv2,[1,2,2,1]) + b_conv2)

            name = "conv3" + main_name
            with tf.variable_scope(name) as scope:
                W_conv3 = self.weight_variable([3, 3, 64, 64])
                b_conv3 = self.bias_variable([64])
                conv3 = tf.nn.relu(self.conv2d(conv2, W_conv3,[1,1,1,1]) + b_conv3)

            h_conv3_shape = conv3.get_shape().as_list()
            #print(h_conv3_shape)
            #print("dimension:",h_conv3_shape[1],h_conv3_shape[2],h_conv3_shape[3])

            name = "fc1" + main_name
            with tf.variable_scope(name) as scope:
                W_fc1 = self.weight_variable([7 * 7 * 64, 512])
                b_fc1 = self.bias_variable([512])
                h_flat = tf.reshape(conv3, [-1, 7*7*64])
                h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

            name = "fc2" + main_name
            with tf.variable_scope(name) as scope:
                W_fc2 = self.weight_variable([512, self.num_actions])
                b_fc2 = self.bias_variable([self.num_actions])

                y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,main_name)
        # return fc final layer and var
        return y_conv, var


    def get_action(self,state):

        #print("** get_action input shape..",state.shape)
        my_feed_dict = {self.x: (state / 255.0) }
        res = self.sess.run( self.y_q_values, feed_dict = my_feed_dict )

        #print("q_value from nn...", res)
        action = np.argmax(res[0])

        if self.epsilon > np.random.random():
            action = np.random.randint(0,self.num_actions)

        epsilon = self.epsilon_update()

        return action, res[0]

    def weight_variable(self,shape):
        with tf.variable_scope("my_weights"):
            initial_value = tf.truncated_normal(shape, stddev=0.1)
            W = tf.get_variable("W",initializer=initial_value)
        return W

    def bias_variable(self,shape):
        with tf.variable_scope("my_bias"):
            initial_value = tf.truncated_normal(shape, 0.0, 0.001)
            b = tf.get_variable("b",initializer=initial_value)
        return b

    def conv2d(self, x, W, strides, name="conv"):
        with tf.variable_scope(name):
            return tf.nn.conv2d(x, W, strides, padding='VALID')

    def build_training_op(self, y_conv, q_network_weights):

        self.a_place = tf.placeholder(tf.int64, [None])
        self.y_place = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(self.a_place, self.num_actions, on_value=1.0, off_value=0.0)
        q_value = tf.reduce_sum(tf.multiply(y_conv, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(self.y_place - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        MOMENTUM = 0.95
        MIN_GRAD = 0.01
        LEARNING_RATE = 0.00025
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return loss, grad_update

    def train(self,mini_batch):

        DECAY_RATE = .99

        states = np.array([each[0] for each in mini_batch])
        actions = np.array([each[1] for each in mini_batch])
        rewards = np.array([each[2] for each in mini_batch])
        dones = np.array([each[3] for each in mini_batch])
        next_states = np.array([each[4] for each in mini_batch])

        batch_size = len(mini_batch)
        targets = np.zeros((batch_size, self.num_actions))

        my_feed_dict = {self.x: (states / 255.0) }
        res = self.sess.run( self.y_q_values, feed_dict = my_feed_dict )

        # copy target matrix eg. 32(batch size) x 6 (num of actions)
        targets = res
        #print(res)
        selected_actions = np.argmax(res, axis=1)

        # rewards normalized with 0 or 1
        # if any points added , it should be 1.
        rewards_binary = (rewards > 0).astype(int)

        #
        my_feed_dict2 = {self.x: (next_states / 255.0) }
        target_q_values2 = self.sess.run( self.y_target, feed_dict = my_feed_dict2 )
        selected_target_actions = np.argmax(target_q_values2, axis=1)

        #print("selected action (init_state)")
        #print(selected_actions)

        #print("selected target action (next_state)")
        #print(selected_target_actions)
        #print("rewards (binary): ")
        #print(rewards_binary)

        #for d in dones:
        #    if d[i] == False:

        y_q = rewards_binary + (1. - dones.astype(int)) * DECAY_RATE * selected_target_actions

        loss_feed_dict ={   self.a_place: actions,
                            self.y_place:y_q,
                            self.x:(states / 255.0)  }
        loss, _ = self.sess.run([self.loss, self.grad_update],
                                feed_dict = loss_feed_dict)

        self.total_loss += loss
        #print(loss)
        #for i in range(batch_size):
        #    init_state = states[i]
        #    targets[i] = self.model.predict(, batch_size = 1)
        #    fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        #    targets[i, a_batch[i]] = r_batch[i]
        #    if d_batch[i] == False:
        #    targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)


def main():

    myAgent = AgentClass(2)

    #y_q , var_q = myAgent.build_network()
    #y_target , var_target = myAgent.build_target()
    #a, y, loss, grad_update = myAgent.build_training_op(y_q, var_q)

    #with tf.Session() as sess:
    #    print("check var of q_network and target_network...")
    #    init_op = tf.group(tf.global_variables_initializer(),
    #                   tf.local_variables_initializer())
    #    sess.run(init_op)
    #    saver = tf.train.Saver()

    #    _q_var = sess.run(var_q)
    #    _t_var = sess.run(var_target)
    #    for idx, v in enumerate( _q_var ):
    #        print(v.shape, _t_var[idx].shape)
    #        assert( v.all() == _t_var[idx].all())

if __name__ == "__main__":
    main()


"""
    def build_network2(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model
"""