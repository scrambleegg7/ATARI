

from ATARI.MCTS import get_module_logger


import numpy as np


mylogger = get_module_logger(__name__)



class A(object):
    def __init__(self):
        pass

    def __str__(self):
        return "A object."

    def process(self):
        mylogger.debug("class A deubg.")



aobject = A()
aobject.process()
mylogger.info("class A : %s", aobject)
