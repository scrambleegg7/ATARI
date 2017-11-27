#import
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym

from MemoryClass import Memory
from tfQnetClass import QNetwork

from time import time

import matplotlib.pyplot as plt




# Network parameters
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

def process(render=False):

    print("CartPole main start..")
    env = gym.make('CartPole-v0')


    # Initialize the simulation
    env.reset()
    # Take one random step to get the pole and cart moving
    state, reward, done, _ = env.step(env.action_space.sample())

    memory = Memory(max_size=memory_size)

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        # Uncomment the line below to watch the simulation
        if render:
            env.render()

        # Make a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, done, _ = env.step(env.action_space.sample())
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

    #memory.checkBuffer()

    return memory, state, env

def training(render = False):

    tf.reset_default_graph()
    mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)


    memory , state, env = process()
    #print(state[:,np.newaxis].transpose().shape)
    #return
    #
    train_episodes = 1000          # max number of episodes to learn from
    max_steps = 200                # max steps in an episode
    gamma = 0.99                   # future reward discount

    # Exploration parameters
    explore_start = 1.0            # exploration probability at start
    explore_stop = 0.01            # minimum exploration probability
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    saver = tf.train.Saver()
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                # Uncomment this next line to watch the training
                if render:
                    env.render()

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = env.action_space.sample()
                else:
                    # Get action from Q-network
                    #feed = {mainQN.inputs_: state.reshape((1, *state.shape))}

                    feed = {mainQN.inputs_: state.reshape((1, state.shape[0] ))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = env.step(env.action_space.sample())

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = (0, 0)

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: targets,
                                               mainQN.actions_: actions})

        saver.save(sess, "checkpoints/cartpole.ckpt")



def main():
    training()


if __name__ == "__main__":
    main()
