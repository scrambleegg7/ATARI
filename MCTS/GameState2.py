#
import random
import math
import hashlib
import logging
import argparse
import numpy as np

from Board import Board


from logging import getLogger, StreamHandler, DEBUG
from logging import Formatter
import logging


class GameState2():

    NUM_TURNS = 10
    GOAL = 0
    PIECE=["X","O"]

    num_moves=9

    def __init__(self, moves=[], board=Board(), turn=0, next_turn=0,test=False):

        self.board=board
        self.turn=turn
        self.next_turn=next_turn
        self.moves=moves

        self.log = getLogger("root")

        self.next_board = Board()

        # check available space when creating Node
        self.initial_space = len( self.board.isfilled() )
        print("GameState2 initial board remaiing space --> %d" % self.initial_space )


    def first_turn(self):
        self.first_turn = np.random.choice([0,1])
        if self.first_turn:
            print("** Human first..")
            self.next_turn = 0
        else:
            print("* AI first..")
            self.next_turn = 1

    def next_state(self):

        # orig board should be copied to next_board
        # check availability next step on next_board
        # if it has space to place, then udate next_board to have new piece
        #
        self.next_board.board = self.board.board.copy()
        sp = self.next_board.isfilled()
        move = np.random.choice(sp)

        turn = self.next_board.board_turn()

        if turn % 2 == 0:
            print("AI turn...%d on board. remaining space %d" % (move,9-turn))
            self.next_board.update(move,self.PIECE[0])
            self.next_turn = 0
        elif turn % 2 == 1:
            print("Human turn...%d on board. remaining space %d" % (move,9-turn))
            self.next_board.update(move,self.PIECE[1])
            self.next_turn = 1

        next=GameState2(moves=self.moves+[ move ], board= self.next_board,turn=(turn +1), next_turn = self.next_turn, test=True)
        return next

    def isWin(self):

        ai_piece = self.PIECE[0]
        hu_piece = self.PIECE[1]
        if self.board.isWin(ai_piece):
            print("AI WIN !")
            return True

        if self.board.isWin(hu_piece):
            print("Human WIN !")
            return True

        #print("Draw - NO WINNER !")
        return False

    def terminal(self):

        if self.turn == 9:
            return True

        return False

    def reward(self):
        #r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
        #return r
        ai_piece = self.PIECE[0]
        hu_piece = self.PIECE[1]

        if self.board.isWin(ai_piece):
            return 20
        elif self.board.isWin(hu_piece):
            return -100
        return -10

    def __hash__(self):
        #return int(hashlib.md5(str(self.board.board).encode('utf-8')).hexdigest(),16)
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        s="CurrentBoardState: %s; turn %d Board:%s"%(self.moves ,self.turn, self.board.board)
        return s
