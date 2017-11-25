
import tensorflow as tf
import numpy as np
import pandas as pd


def inference(x_ph, num_states, h1, h2, num_actions):


    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([NUM_INPUT, NUM_HIDDEN1], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_HIDDEN1], dtype=tf.float32), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(x_ph, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN1, NUM_HIDDEN2], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_HIDDEN2], dtype=tf.float32), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN2, NUM_OUTPUT], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_OUTPUT], dtype=tf.float32), name='biases')
        y = tf.matmul(hidden2, weights) + biases

    return y

def loss(y, y_ph):
    return tf.reduce_mean(tf.nn.l2_loss((y - y_ph)))

def optimize(loss,lr):
    LEARNING_RATE = lr
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_step = optimizer.minimize(loss)
    return train_step
