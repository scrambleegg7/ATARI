#
import random
import math
import hashlib
import logging
import argparse
import numpy as np


class Board(object):
    def __init__(self, board=np.array( [" "] * 10 )):
        self.board = board # board is numpy array

    def update(self,move,piece):
        self.board[move] = piece

    def isAvailable(self,move):
        s = self.isfilled()
        if move in s:
            return True
        else:
            return False

    def board_copy(self):
        return self.board.copy()

    def isfilled(self):
        space = np.where(self.board == " ")[0]
        return space[1:]

    def board_turn(self):
        space = self.isfilled()
        #print("** available space. (Board Class)", space)
        if len(space) > 0:
            turn = 9- len(space)
        if len(space) == 0:
            turn = 9
        return turn

    def __repr__(self):
        print('   |   |')
        print(' ' + self.board[7] + ' | ' + self.board[8] + ' | ' + self.board[9])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + self.board[4] + ' | ' + self.board[5] + ' | ' + self.board[6])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + self.board[1] + ' | ' + self.board[2] + ' | ' + self.board[3])
        print('   |   |')

        turn = self.board_turn()
        s="Current Board State. (fm:Board Class): turn %d"%(turn)
        return s

    def isWin(self,le):
        # bo - input board style
        # le - letter on board from player's piece
        bo = self.board.copy()
        return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
        (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
        (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
        (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
        (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
        (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
        (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
        (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal
