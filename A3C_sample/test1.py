#
import numpy as np
import tensorflow  as tf

import os
import sys, getopt
import multiprocessing
import threading
import numpy as np
from time import time, sleep, gmtime, strftime
import gym
import queue
import random
from agent import Agent
import logging

# FLAGS
T_MAX = 100000000
NUM_THREADS = 8
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
VERBOSE_EVERY = 40000
TESTING = False

I_ASYNC_UPDATE = 5

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
random.seed(100)

def async_trainer(agent, env, sess, thread_idx, T_queue, summary, saver,
    save_path):
    print "Training thread", thread_idx
    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    last_verbose = T
    last_time = time()
    last_target_update = T

    terminal = True
    while T < T_MAX:
        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            terminal = False
            state = env.reset()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # Save the current state
            batch_states.append(state)

            # Choose an action randomly according to the policy
            # probabilities. We do this anyway to prevent us having to compute
            # the baseline value separately.
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal.
            state, reward, terminal, _ = env.step(action_idx)

            # Update counters
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)

            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value[0])

        target_value = 0
        # If the last state was terminal, just put R = 0. Else we want the
        # estimated value of the last state.
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)
        # Reverse the batch target values, so they are in the correct order
        # again.
        batch_target_values.reverse()

        # Test batch targets
        if TESTING:
            temp_rewards = batch_rewards + [last_R]
            test_batch_target_values = []
            for j in range(len(batch_rewards)):
                test_batch_target_values.append(discount(temp_rewards[j:], DISCOUNT_FACTOR))
            if not test_equals(batch_target_values, test_batch_target_values,
                1e-5):
                print "Assertion failed"
                print last_R
                print batch_rewards
                print batch_target_values
                print test_batch_target_values

        # Compute the estimated value of each state
        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)

        # Apply asynchronous gradient update
        agent.train(np.vstack(batch_states), batch_actions, batch_target_values,
        batch_advantages)

    global training_finished
    training_finished = True

def main():

    logging.debug('session start ...')

    env =

    with tf.Session() as sess:
            agent = Agent(session=sess,
            action_size=3, model='mnih',
            optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))



if __name__ == "__main__":
    main()
