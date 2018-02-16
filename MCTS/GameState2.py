#
import random
import math
import hashlib
import argparse
import numpy as np
import copy

from Board import Board

from ATARI.MCTS import get_module_logger

mylogger = get_module_logger(__name__)

class GameState2():

    NUM_TURNS = 10
    PIECE=["X","O"]
    player = {}
    player["a"] = "X"
    player["h"] = "O"

    num_moves=9

    def __init__(self, moves=[], board=Board(), turn=0, next_turn="a",check=False):

        self.board=board
        self.turn=turn
        self.next_turn=next_turn
        self.moves=moves

        self.check = check
        self.opposit_predict_win = None

        self.next_board = Board()

        # check available space when creating Node
        self.initial_space = len( self.board.isfilled() )
        mylogger.debug("initialize <GameState2>, board remaiing space --> %s Next turn:%s" % (self.board.isfilled(), self.next_turn) )
        mylogger.debug("initial piece move", self.moves)
        mylogger.debug("initial piece location", self.board.board)

    def first_turn(self):
        self.first_turn = np.random.choice([0,1])
        if self.first_turn:
            mylogger.debug("** Human first..")
            self.next_turn = 0
        else:
            mylogger.debug("* AI first..")
            self.next_turn = 1


    def check_play(self,board):

        check_board = copy.deepcopy(board)
        available_sp = check_board.isfilled()

        piece = self.player[self.next_turn]
        for s in available_sp:

            check_board.update(s,piece)
            if check_board.isWin(piece):
                mylogger.debug(" ** check !! **")
                return True

        return False

    def available(self,sp):


        # for checking ..
        for s in sp:
            temp_board = copy.deepcopy(self.board)
            opposit_board = copy.deepcopy(self.board)

            ai_piece = self.player["a"]
            hu_piece = self.player["h"]

            temp_board.update(s,ai_piece)
            opposit_board.update(s,hu_piece)

            #mylogger.debug("temp_board ..",temp_board.board)

            if temp_board.isWin(ai_piece):
                mylogger.debug("%d is winning move for ai. (available)" % s)
                self.opposit_predict_win = True
                return s
            if opposit_board.isWin(hu_piece):
                mylogger.debug("%d is winning move for human. (available)" % s)
                self.opposit_predict_win = True
                return s

        return np.random.choice(sp)

    def switchTurn(self):
        #
        # do not override any original next turn
        #
        # new next_turn should be passed to new object of GameState2 ..
        #
        if self.next_turn == "a": # AI
            return "h" # Human
        else:
            return "a" # AI

    def next_state_1(self):

        # orig board should be copied to next_board
        # check availability next step on next_board
        # if it has space to place, then udate next_board to have new piece
        #
        self.next_board.board = self.board.board.copy()
        sp = self.next_board.isfilled()

        move = np.random.choice(sp)

        turn = self.next_board.board_turn()

        if turn % 2 == 0:
            mylogger.debug("AI turn...%d on board. remaining space %d" % (move,9-turn))
            self.next_board.update(move,self.PIECE[0])
            self.next_turn = 0
        elif turn % 2 == 1:
            mylogger.debug("Human turn...%d on board. remaining space %d" % (move,9-turn))
            self.next_board.update(move,self.PIECE[1])
            self.next_turn = 1

        next=GameState2(moves=self.moves+[ move ], board= self.next_board,turn=(turn +1), next_turn = self.next_turn, test=True)
        return next

    def next_state(self):

        # orig board should be copied to next_board
        # check availability next step on next_board
        # if it has space to place, then udate next_board to have new piece
        #
        self.next_board = copy.deepcopy(self.board)
        available_sp = self.next_board.isfilled()

        move = self.available(available_sp)
        remaining_sp = len(available_sp) - 1

        if self.next_turn == "a": # AI
            self.next_board.update(move,self.PIECE[0])
            mylogger.debug("AI turn...%d on board. remaining space %d" % (move,remaining_sp))

        elif self.next_turn == "h": # Human
            self.next_board.update(move,self.PIECE[1])
            mylogger.debug("Human turn...%d on board. remaining space %d" % (move,remaining_sp))

        check = self.check_play(self.next_board)
        next_turn = self.switchTurn() # update next_turn..

        mylogger.debug("* build new GameState2.. *")
        next=GameState2(moves=self.moves+[ move ], board= self.next_board,turn=(self.turn +1), next_turn = next_turn, check=check)
        return next

    def isWin(self):

        ai_piece = self.PIECE[0]
        hu_piece = self.PIECE[1]
        if self.board.isWin(ai_piece):
            mylogger.debug("AI WIN !")
            return True

        if self.board.isWin(hu_piece):
            mylogger.debug("Human WIN !")
            return True

        #mylogger.debug("Draw - NO WINNER !")
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
            return -20
        return 0

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
