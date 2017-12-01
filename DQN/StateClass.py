#
import numpy as np
from skimage.transform import resize

from env import setEnv


def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class SteteClass(object):

    def __init__(self, env):

        self.env = env
        self.observation = self.env.reset()

        obs_shape = self.observation
        if len(obs_shape) < 3:
            print("no image data")

        self.image_buffer = []

    def render(self):
        self.env.render()

    def envReset(self):
        obs = self.env.reset()

    def add_buffer(self,obs):

        self.image_buffer.append(obs)

    def clearImageBuffer(self):
        self.image_buffer = []

    def add_frame(self,action,num_frames):

        self.clearImageBuffer()
        rewards = 0
        for i in range(num_frames):
            tmp_obs, tmp_reward, tmp_done, _ = self.env.step(action)
            rewards += tmp_reward
            self.add_buffer(tmp_obs)

        return tmp_obs, rewards, tmp_done

    def initial_buffer(self):

        for i in range(3):
            #
            # observation for non action ....
            #
            st, _, _, _ = self.env.step(0)
            self.image_buffer.append(st)

    def initial_buffer2(self):

        self.clearImageBuffer()
        observation = self.envReset()

        #
        # forward game screen with random number iteration
        # action -- nothing to move any cursor
        #
        for i in range(np.random.randint(1,10)):
            #
            # observation for non action ....
            #
            last_observation = observation
            observation, _, _, _ = self.env.step(0)
            #self.image_buffer.append(st)

        processed_image = self.preprocess(observation,last_observation)
        for _ in range(3):
            self.image_buffer.append(processed_image)

    def preprocess(self, observation, last_observation):

        processed_observation = np.maximum(observation, last_observation)
        #processed_observation = np.uint8(resize(rgb2gray(processed_observation), (84, 84)) * 255)
        # rgb image 84x84
        return processed_observation

    def convertAndConcatenateBuffer(self):

        buffers = []
        for buff in self.image_buffer:
            gr_resize = self.gray_resize(self.convertRGB(buff),84,84)
            buffers.append( gr_resize )

        black_buffer = [x for x in buffers]
        b = np.stack(black_buffer,axis=0)
        b = np.transpose(b, (1,2,0))
        #b = np.concatenate(black_buffer,axis=2)
        #return b[np.newaxis,:,:,:]
        return b

    def convertRGB(self,img):
        g = rgb2gray(img)
        return g

    def gray_resize(self, img, new_r, new_c):
        resize_gray = resize( img,  (new_r,new_c) )
        return resize_gray
