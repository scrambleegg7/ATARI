
import gym
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt


from MemoryClass import Memory
from StateClass import SteteClass
#from env import setEnv
from AgentClass import AgentClass

from PIL import Image


myMemory = Memory(max_size=10)


x = range(20)

for item in x:

    myMemory.add(item)


print( myMemory.checkBuffer() )
