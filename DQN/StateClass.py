#
import numpy as np
from skimage.transform import resize

from env import setBreakEnv


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

    def add_buffer(self,predict_action):

        pass



    def initial_buffer(self):

        for i in range(3):
            st, _, _, _ = self.env.step(0)
            self.image_buffer.append(st)

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