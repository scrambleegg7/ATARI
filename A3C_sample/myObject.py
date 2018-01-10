#
import numpy as np
import pandas as pd

import logging

class myObject(object):

    def __init__(self, test=False):
        self.test = test
        self.counter = 0

    def increment(self):
        logging.debug("object increment ....")
        self.counter += 1
        c = self.getCounter()
        logging.debug("after increment .. %d", c)

    #@property
    def getCounter(self):
        return self.counter
