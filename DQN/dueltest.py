#
import tensorflow as tf
import numypy as np
import pandas as pd

x = tf.placeholder(np.float32, [None, 21168])

x_image = tf.reshape(x, [-1,84,84,3])



class dueltestclass(object):

    def __init__(self,test=False):
        self.test = test

    def build_network(self,main_name="Q",reuse=False):
        # reuse is used to share weight and other values..

        with tf.variable_scope(main_name) as scope:

            x_image = tf.reshape(self.x, [-1, 84, 84, STATE_LENGTH])

            name = "conv1"
            with tf.variable_scope(name) as scope:
                W_conv1 = self.weight_variable([8, 8, STATE_LENGTH, 32])
                b_conv1 = self.bias_variable([32])
                #conv1 = tf.nn.relu(self.conv2d(self.x, W_conv1,[1,4,4,1] ) + b_conv1)
                conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1,[1,4,4,1] ) + b_conv1)

            name = "conv2"
            with tf.variable_scope(name) as scope:
                W_conv2 = self.weight_variable([4, 4, 32, 64],reuse)
                b_conv2 = self.bias_variable([64],reuse)
                conv2 = tf.nn.relu(self.conv2d(conv1, W_conv2,[1,2,2,1]) + b_conv2)

            name = "conv3"
            with tf.variable_scope(name) as scope:
                W_conv3 = self.weight_variable([3, 3, 64, 64],reuse)
                b_conv3 = self.bias_variable([64],reuse)
                conv3 = tf.nn.relu(self.conv2d(conv2, W_conv3,[1,1,1,1]) + b_conv3)

            h_conv3_shape = conv3.get_shape().as_list()
            #print(h_conv3_shape)
            #print("dimension:",h_conv3_shape[1],h_conv3_shape[2],h_conv3_shape[3])

            name = "fc1"
            with tf.variable_scope(name) as scope:
                W_fc1 = self.weight_variable([7 * 7 * 64, 512],reuse)
                b_fc1 = self.bias_variable([512],reuse)
                h_flat = tf.reshape(conv3, [-1, 7*7*64])
                h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

            name = "fc2"
            with tf.variable_scope(name) as scope:
                W_fc2 = self.weight_variable([512, self.num_actions],reuse)
                b_fc2 = self.bias_variable([self.num_actions],reuse)

                y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,main_name)
        # return fc final layer and var
        return y_conv, var
