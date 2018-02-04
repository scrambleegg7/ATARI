#
import numpy as np
import math
import logging
import hashlib

from uct import UCTSEARCH
from node import Node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GameState')

NUM_TURNS = 9



def isWin(bo,le):
    # bo - input board style
    # le - letter on board from player's piece
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
    (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
    (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
    (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
    (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
    (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal


class Player(object):

    def __init__(self, first):
        self.first = first

    def isSpaceFree(self, board):
        b = np.array(board)
        space = np.where( b  == " "  )
        return space[0][1:]

    def isAvailable(self, board, move):

        m = int(move)

        space = self.isSpaceFree(board)
        #logging.debug("move:%s, available space:%s",move,space)

        if m in space:
            return True
        else:
            return False

class huPlayer(Player):

    def __init__(self,first=True):
        super(huPlayer, self).__init__(first)
        self.piece = "X"


    def randomPlay(self,board):

        space = self.isSpaceFree(board)
        r = np.random.choice(space)
        return r

    def getPlayerMove(self, board):

        # Let the player type in their move.
        move = ' '
        while move not in '1 2 3 4 5 6 7 8 9'.split() or not self.isAvailable(board, move):
            print('What is your next move? (1-9)')
            move = input()

        return int(move)



class aiPlayer(Player):

    def __init__(self,first=False):
        super(aiPlayer,self).__init__(first)
        self.piece = "O"

    def playerMove(self):
        pass

    def randomPlay(self,board):

        space = self.isSpaceFree(board)
        r = np.random.choice(space)
        logging.info("available space for ai...%s ",space)
        logging.info("ai selected %d..", r)
        return r

    def simulaton(self,board):

        space = self.isSpaceFree(board)
        NUM_TURNS = len(space)

        for l in range(NUM_TURNS):
            pass


class GameState(object):

    def __init__(self, board = [' '] * 10,turn = 0, next_turn=0):

        self.board = board
        self.turn = turn
        self.next_turn = next_turn
        self.num_moves = 10

    def getBoard(self):

        return self.board

    def makeMove(self,move,le):

        self.board[move] = le

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

        s="CurrentState ...: turn %d"%(self.turn)
        return s

    def reward(self, le):
    # Given a board and a player’s letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don’t have to type as much.
        r = self.isWin(le)
        if r:
            return 10
        else:
            return -10


    def terminal(self):

        if isWin(self.board,self.aiPlayer.piece):
            return True
        elif isWin(self.board,self.huPlayer.piece):
            return True
        elif self.turn == NUM_TURNS:
            return True
        else:
            print("sorry no winner from terminal..")
            return False

    def aiPlayerAction(self):
        move = self.aiPlayer.randomPlay(self.board)
        piece = self.aiPlayer.piece
        self.makeMove(move,piece)
        #print(move)

    def huPlayerAction(self): # from human player with manual entry with number...
        move = self.huPlayer.randomPlay(self.board)
        piece = self.huPlayer.piece
        self.makeMove(move,piece)


    def huPlayerActionManual(self): # from human player with manual entry with number...
        move = self.huPlayer.getPlayerMove(self.board)
        piece = self.huPlayer.piece
        self.makeMove(move,piece)

    def next_state_simulation(self):

        if self.next_turn == 0:
            self.aiPlayerAction()
            self.next_turn = 1
        else:
            self.huPlayerAction()
            self.next_turn = 0

        self.turn += 1

        next = GameState(self.board,turn=1)
        return next

    def next_state(self):

        if self.aiPlayer.first:
            self.aiPlayerAction()
            #print(self)
            self.huPlayerAction()
            #print(self)
        else:
            self.huPlayerAction()
            #print(self)
            self.aiPlayerAction()
            #print(self)

        logging.info("currnet board for ai...%s ",self.board)

        next = GameState(self.board)
        return next

    def next_state_manual(self):

        if self.aiPlayer.first:
            self.aiPlayerAction()
            print(self)
            self.huPlayerAction()
            print(self)
        else:
            self.huPlayerAction()
            print(self)
            self.aiPlayerAction()
            print(self)

        logging.info("currnet board for ai...%s ",self.board)

    def firstTurn(self):
        c = np.random.choice((0,1))
        if c == 0: # aiPlayer is first
            self.aiPlayer = aiPlayer(first=True)
            self.huPlayer = huPlayer(first=False)
            print("* AI first turn.....")
            self.next_turn = 1
        else:
            self.aiPlayer = aiPlayer(first=False)
            self.huPlayer = huPlayer(first=True)
            print("** human first turn.....")
            self.next_turn = 0

    def __hash__(self):

	       return int(hashlib.md5(str(self.current)).hexdigest(),16)

    def __eq__(self,other):
    	if hash(self)==hash(other):
    	    return True
    	return False



def main():

    print("## Game Logic .. for MCTS ...")

    gs = GameState()
    gs.firstTurn() # decide which turn place piece first

    num_sims = 100
    myGamePlay = True

    #while myGamePlay:


    current_node = Node(gs)

    for l in range(NUM_TURNS):
        logging.info("LEVEL : %d", l)
        #gs.next_state()


        current_node=UCTSEARCH(num_sims/(l+1),current_node)


if __name__ == "__main__":
    main()
