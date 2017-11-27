#
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize

from AgentClass import AgentClass

from MemoryClass import Memory

ENV_NAME = "Breakout-v0"
env = gym.make(ENV_NAME)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def initial_action():

    observation = env.reset()
    rewards = []
    rand_actions = []

    #for _ in range( np.random.randint(1, 50) ):
    #    #env.render()
    #    last_observation = observation
    #    observation, _, _, _ = env.step(0) # take a random action

    #state = np.maximum(observation,last_observation)

    # initial screen
    state = observation
    state_gray = rgb2gray(state)
    resize_gray = resize( state_gray, (84,84) ).astype(np.int8)

    # Converts the list of NUM_FRAMES images in the process buffer
    # into one training sample
    black_buffer = [resize_gray for _ in range(4) ]
    black_buffer = [x[:, :, np.newaxis] for x in black_buffer]
    return np.concatenate(black_buffer, axis=2)

def BasicScenario():

    for _ in range(400):
        #env.render()
        r_action = env.action_space.sample()
        state, reward, done, info = env.step(r_action) # take a random action

        rewards.append(reward)
        rand_actions.append(r_action)
        if done:
            print("done......")
            print("reward %d state %s" % (reward, state.shape))
            rewards = []
            env.reset()

    return np.sum(rewards)
    #plt.plot(range(len(rewards)), np.cumsum(rewards), c="r")
    #plt.show()

def train():


    #print(x_image.shape)
    #plt.imshow(x_image)
    #plt.show()

    myAgent = AgentClass(2)

    # initialize Q Network
    y_q , var_q = myAgent.build_network()

    # intialize Target Network
    y_target , var_target = myAgent.build_target()

    # get traing loss
    a, y, loss, grad_update = myAgent.build_training_op(y_q, var_q)

    epsilon = 1.0
    EPSILON_DECAY = 300000
    FINAL_EPS = 0.1

    with tf.Session() as sess:
        print("check var of q_network and target_network...")
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        episodes = 10

        for i_frames in range(episodes): # num of frames

            # initial screen
            init_state = initial_action()

            if epsilon >= np.random.random():
                action = env.action_space.sample()
                #action = np.random.choice(2,1)
                print("from random:", action)
            else:
                a = y_q.eval(feed_dict = {x:init_state})
                action = np.argmax(a)
                print("from model:",action)
            repeated_action = action

            if epsilon > FINAL_EPS:
                epsilon -= (epsilon - FINAL_EPS) / EPSILON_DECAY

            init_state = initial_action()


            observation, reward, done, _ = env.step(action)


def main():

    train()


    #rewards_hist = []
    #for i in range(1000): # number of episodes
    #    reward_sum = BasicScenario()
    #    rewards_hist.append(reward_sum)

    #print("avg. rewards from random action ...", np.mean(rewards_hist))

    #plt.plot(range(len( rewards_hist  )), rewards_hist )
    #plt.show()


if __name__ == "__main__":
    main()
