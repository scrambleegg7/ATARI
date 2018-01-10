from scipy.misc import imresize
import gym
import numpy as np
import random

from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity


class CustomGym:
    def __init__(self, game_name, skip_actions=4, num_frames=4, w=84, h=84):
        self.env = gym.make(game_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h
        #if game_name == 'SpaceInvaders-v0':
        #    self.action_space = [1,2,3] # For space invaders
        #elif game_name == 'Pong-v0':
        #    self.action_space = [1,2,3]
        #elif game_name == 'Breakout-v0':
        #    self.action_space = [1,4,5]
        #else:
            # Use the actions specified by Open AI. Sometimes this has more
            # actions than we want, and some actions do the same thing.
        self.action_space = range(self.env.action_space.n)

        self.action_size = len(self.action_space)
        self.observation_shape = self.env.observation_space.shape

        self.state = None
        self.game_name = game_name


    @property
    def state_shape(self):
        return [self.w, self.h, self.num_frames]

    @property
    def action_dim(self):
        return self.env.action_space.n

    def preprocess(self, obs, is_start=False):
        grayscale = obs.astype('float32').mean(2)
        s = imresize(grayscale, (self.w, self.h)).astype('float32') * (1.0/255.0)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)
        return self.state

    def render(self):
        self.env.render()

    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    def get_initial_state(self, observation):

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
        init_image /= 255.0
        # should be changed W x H x C format
        return init_image

    def reset_1(self):
        observation = self.env.reset()
        for _ in range(np.random.randint(1,10)):
            last_observation = observation
            observation, reward, done, info = self.env.step(0)

        #print("** initial state image ...")
        state_image = self.get_initial_state(observation)
        self.state_image = state_image
        state_image = state_image.reshape(1, state_image.shape[0], state_image.shape[1], 4)

        return state_image

    def step_1(self, action_idx):

        # 1. input : action index
        # 2. get single converted 84x84 BW image
        # 3. append S(previous) to single image -> 4 single images consolidated.
        #    a. image index : 0 1 2 3
        #    b. image index : 1 2 3
        #    c. image index : 1 2 3 + new single image from preprocess_1
        # 4. devide 4 consolidated images with 255
        # 5. convert [0 .. 1]
        # 6. reshape for tensorflow format 1 x 84 x 84 x 4


        observation, reward, done, info = self.env.step(action_idx)

        processed_image = self.preprocess_1(observation)

        next_image = np.append(self.state_image[:, :, 1:], processed_image[:,:,np.newaxis], axis=2)
        #state_image /= 255.0
        next_image /= 255.0

        next_image = next_image.reshape(1,next_image.shape[0],next_image.shape[1],4)

        return next_image, reward, done, info

    def preprocess_1(self,observation):

        #
        # 1. input : observation image from step or reset_1
        # 2. convert B&W
        # 3. resize 84 x 84
        # 4. normalize 0 , 255
        # 5. divide 255 if necessary.
        #
        #processed_observation = np.maximum(observation, last_observation)
        obs_image = rgb2gray(observation)
        obs_image = resize(obs_image, (84,84),mode="constant")
        obs_image = rescale_intensity(obs_image,out_range=(0,255))

        return obs_image

    def step(self, action_idx):
        action = self.action_space[action_idx]
        accum_reward = 0
        prev_s = None
        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break
            prev_s = s
        # Takes maximum value for each pixel value over the current and previous
        # frame. Used to get round Atari sprites flickering (Mnih et al. (2015))
        if self.game_name == 'SpaceInvaders-v0' and prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info


def main():

    customgym = CustomGym('SpaceInvaders-v0')
    s = customgym.reset_1()
    print(np.min(s),np.max(s))
    print(s.shape)

    s_, r, done, _ = customgym.step_1(0)
    print(s.shape)
    print(np.min(s_),np.max(s_))

if __name__ == "__main__":
    main()
