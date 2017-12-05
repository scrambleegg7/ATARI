
import gym
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt


from MemoryClass import Memory
from StateClass import SteteClass
#from env import setEnv
from AgentClass import AgentClass

from PIL import Image

ENV_NAME = 'SpaceInvaders-v0'
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME

env = gym.make(ENV_NAME)
myAgent = AgentClass(env.action_space.n)

observation = env.reset()

memory_size = 100
memory = Memory(max_size=memory_size)


def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def init_resize(observation):
    obs_resize_bw_image = np.uint8(resize(rgb2gray(observation), (84, 84)) * 255)
    state = [obs_resize_bw_image for _ in range(3)]
    stacked_image = np.stack(state, axis=0)
    init_image = np.transpose(stacked_image,(1,2,0))

    return init_image

def obs_resize(observation):
    obs_resize_bw_image = np.uint8(resize(rgb2gray(observation), (84, 84)) * 255)

    return obs_resize_bw_image

state_image = init_resize(observation)



#state_image = np.uint8(resize(rgb2gray(observation), (84, 84)) * 255)
actions = []
state_images = []
next_images = []
rewards = []
dones = []

for _ in range(100):

    action, q_value = myAgent.get_action(state_image)
    actions.append(action)

    observation, reward, done, info = env.step(action)
    next_obs_image = obs_resize(observation)
    next_obs_image = np.array(next_obs_image)

    next_image = np.append(state_image[:, :, 1:], next_obs_image[:,:,np.newaxis], axis=2)

    state_images.append( state_image)
    next_images.append(next_image)
    rewards.append(reward)
    dones.append(done)

print("test actions....")
print(actions)
print("test rewards....")
print(rewards)

DECAY_RATE = 0.99

actions = np.array(actions)
next_images =  np.array(next_images)
state_images =  np.array(state_images)
dones = np.array(dones)
q_value = myAgent.get_q_value(state_images)
#print(q_value)


q_targets = myAgent.get_q_target_value(next_images)
selected_target_actions = np.max(q_targets,axis=1)
#selected_target_actions = np.max(q_targets, axis=1)
print(selected_target_actions)


y_q = rewards + (1. - dones.astype(int)) * DECAY_RATE * selected_target_actions

print(y_q)


loss_feed_dict ={   myAgent.a_place: actions,
                    myAgent.y_place:y_q,
                    myAgent.x:(state_images / 255.0)  }
loss, _ = myAgent.sess.run([myAgent.loss, myAgent.grad_update],
                        feed_dict = loss_feed_dict)

print(loss)

#print(y_q)
