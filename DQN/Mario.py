#
import gym

class Mario(object):

    def __init__(self):

        self.env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
