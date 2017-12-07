
import gym
import numpy as np

from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt

from MemoryClass import Memory
from StateClass import SteteClass
#from env import setEnv
from AgentClass_v3 import AgentClass

# parameters ...
#
num_consecutive_iterations = 100
num_episodes = 500

initial_training = 20000 # start point to train the data
memory_size = 30000
memory = Memory(max_size=memory_size)

MINIBATCH_SIZE = 32

ENV_NAME = 'SpaceInvaders-v0'
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME

env = gym.make(ENV_NAME)

STATE_LENGTH=4
myAgent = AgentClass(env.action_space.n,STATE_LENGTH)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_initial_state(observation, last_observation):

    init_image = rgb2gray(observation)
    init_image = resize(init_image, (84,84),mode="constant")
    init_image = rescale_intensity(init_image,out_range=(0,255))

    #processed_observation = np.maximum(observation, last_observation)
    #processed_observation = np.uint8(resize(rgb2gray(last_observation), (84, 84)) * 255)
    state = [init_image for _ in range(4)]
    stacked_image = np.stack(state, axis=0)

    #
    #  one layer equal to one game screen image ...
    #

    #  layer 0 --> game image (no action)
    #  layer 1 --> game image (no action)
    #  layer 2 --> game image (no action)
    #
    init_image = np.transpose(stacked_image,(1,2,0))
    # should be changed W x H x C format
    return init_image

def preprocess(observation, last_observation):
    #processed_observation = np.maximum(observation, last_observation)
    obs_image = rgb2gray(observation)
    obs_image = resize(obs_image, (84,84),mode="constant")
    obs_image = rescale_intensity(obs_image,out_range=(0,255))

    return obs_image
    #return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def trainProc():

    print("SpaceInvador action number..",  env.action_space.n )

    global_steps = 0

    mspacman_mean = np.array([270,164,74]).mean()
    print("* pacman mean ...",mspacman_mean)

    for episode in range(num_episodes):
        #
        observation = env.reset()
        #img = observation[1:176:2,::2]
        #print(img.shape)
        #plt.imshow(img)
        #plt.show()
        #break

        for _ in range(np.random.randint(1,10)):
            last_observation = observation
            observation, reward, done, info = env.step(0)

            #print( np.min(observation.ravel()), np.max(observation.ravel()) )

        #print("** making initial state image...")
        state_image = get_initial_state(observation, last_observation)

        episode_reward = 0
        episode_max_q_value = 0
        done = False
        step = 0

        while not done:
            #env.render()

            last_observation = observation
            #action = np.random.randint(0,env.action_space.n)

            # input : state_image
            action, q_value = myAgent.get_action(state_image, global_steps)
            observation, reward, done, info = env.step(action)

            episode_reward += reward
            processed_image = preprocess(observation, last_observation)

            #
            # forward 1 frame and make 3 layer image..
            #
            #state_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
            #memory.add((state_image, action, rewards, done, processed_image))
            state_image, max_q_value = run(global_steps,state_image, action, reward, done, processed_image)

            global_steps += 1
            episode_max_q_value += max_q_value

            if done:
                #
                # finish n episodes
                #
                print("episode:{}  memory length:{}".format(episode,memory.checklength()) )
                if memory.checklength() > (memory_size-1):
                    #total_loss = myAgent.getTotalloss()
                    #myAgent.write_tfValueLog(step,episode,episode_reward,episode_max_q_value)

                    print('%d Episode finished after %f steps / mean %f' %
                                (episode, step + 1, episode_reward / (step+1)) )
                    print("   Global Step", global_steps )
                    print("   avg. training_loss ..", myAgent.getTotalloss() / step)
                    #print("   avg. max_q_value ..", episode_max_q_value / step)

            step += 1

def run(global_steps,state_image,action,rewards,done,processed_image):

    next_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
    state_image /= 255.0
    next_image /= 255.0
    memory.add((state_image, action, rewards, done, next_image))

    q_value_state_image = myAgent.get_q_value(state_image)
    max_q_value = np.max(q_value_state_image)

    # default training_loss :
    #print( memory.checklength() )
    if global_steps > initial_training and global_steps % 4 == 0:
        mini_batch = memory.sample(batch_size=MINIBATCH_SIZE)
        myAgent.train(mini_batch, global_steps)

    if global_steps % 10000 == 0 and global_steps > 0 and global_steps > (initial_training + 1):
        print("    training counter...", global_steps)
        print("")
        print("** copy Qnetwork w/b --> target network w/b ...")
        print("")
        myAgent.copyTargetQNetwork()

    if global_steps % 20000 == 0 and global_steps > 0:
        myAgent.saver.save(myAgent.sess,"tfmodel/dqnMain",global_step=global_steps)
        print("**    model saved.. ")


    return next_image, max_q_value

def main():
    trainProc()

if __name__ == "__main__":
    main()
