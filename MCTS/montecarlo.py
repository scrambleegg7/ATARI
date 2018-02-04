#
import numpy as np
import math
import hashlib
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MonteCarlo')

class MonteCarlo(object):


    def __init__(self, board, test=False):

        self.test = test
        self.board = board

        self.com_budget = 1000 // simulaton number

    def __str__(self):
        logging.debug("string function...")

    def __repr__(self):

        logging.debug("repr  ....")

    def play_game(self):
        pass


    def run_siulation(self):

        i = 0
        for _ in range(self.com_budget):



            n = self.treePolicy()



            delta = self.defaultPolicy()

            i += 1

            self.backup()


        pass

    def treePolicy(self):

        return False

    def defaultPolicy(self):

        return False

    def backup(self):

        pass
