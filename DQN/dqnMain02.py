
import gym
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt


from MemoryClass import Memory
from StateClass import SteteClass
#from env import setEnv
from AgentClass import AgentClass



# parameters ...
#
goal_average_steps = 195
max_number_of_steps = 2000
num_consecutive_iterations = 100
num_episodes = 100


memory_size = 500
memory = Memory(max_size=memory_size)

MINIBATCH_SIZE = 32

ENV_NAME = 'SpaceInvaders-v0'
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME

env = gym.make(ENV_NAME)
myAgent = AgentClass(env.action_space.n)


def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_initial_state(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
    state = [processed_observation for _ in xrange(3)]
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
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
    return processed_observation
    #return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))

def trainProc():

    print("SpaceInvador action number..",  env.action_space.n )

    last_time_steps = np.zeros(num_consecutive_iterations)


    for episode in range(num_episodes):
        #
        observation = env.reset()

        for _ in range(np.random.randint(1,10)):
            last_observation = observation
            observation, reward, done, info = env.step(0)

        #print("** making initial state image...")
        state_image = get_initial_state(observation, last_observation)
        #plt.imshow(image[1:,:,:])
        #plt.show()
        #break

        episode_reward = 0
        episode_max_q_value = 0
        done = False
        #for t in range(max_number_of_steps):
        step = 0
        while not done:
            #env.render()

            last_observation = observation
            #action = np.random.randint(0,env.action_space.n)
            action, q_value = myAgent.get_action(state_image)
            observation, reward, done, info = env.step(action)

            episode_reward += reward
            processed_image = preprocess(observation, last_observation)

            #
            # forward 1 frame and make 3 layer image..
            #
            #state_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
            #memory.add((state_image, action, rewards, done, processed_image))
            state_image, max_q_value = run(step, state_image, action, reward, done, processed_image)

            episode_max_q_value += max_q_value

            if done:
                #
                # finish n episodes
                #

                print('%d Episode finished after %f steps / mean %f' %
                            (episode, step + 1, episode_reward / (step+1)) )
                print("   avg. training_loss ..", myAgent.getTotalloss() / step)
                print("   avg. max_q_value ..", episode_max_q_value / step)

                myAgent.resetTotalloss()

            step += 1


def run(step, state_image, action, rewards, done, processed_image):

    next_image = np.append(state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
    memory.add((state_image, action, rewards, done, next_image))

    q_value_state_image = myAgent.get_q_value(state_image)
    max_q_value = np.max(q_value_state_image)

    # default training_loss :
    #print( memory.checklength() )
    if memory.checklength() > (memory_size-1) and step % 3 == 1:
        mini_batch = memory.sample(batch_size=MINIBATCH_SIZE)
        myAgent.train(mini_batch)

    if done:
        pass

    return next_image, max_q_value

def main():
    trainProc()

if __name__ == "__main__":
    main()

#print(state_image.shape)
