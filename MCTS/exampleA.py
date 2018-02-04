#
import numpy as np
import math

class Node(object):

    def __init__(self):

        self.child = Node()
        self.node_reward = 0

        self.child_array = []


    def add_child_node(self,node):

        child = node(self.child, self)
        self.child_array.append(child)

    def genNode(self):
        pass

    def checkChildArray(self):
        return len(self.child_array) != 0

    def getState(self):




class Board(object):

    def __init__(self):

        self.playerA = "A"
        self.playerB = "B"

    

class Tree(object):

    def __init__(self):

        self.root = Node()
        self.state = None

    def getState(self):
        return self.state



class State(object):

    def __init__(self):

        self.board = None

        self.playNo = 0
        self.visitCount = 0
        self.winScore = 0

    def setBoard(self,board):
        self.board = board

    def setPlayNo(self, opponent):
        self.playNo = opponent

    def getAvailableStates(self):
        pass

    def randomPlay(self):
        pass

class MonteCarloSearch(object):

    def __init__(self):

        self.winScore = 10
        self.level = 0
        self.opponent = 1

    def findNextMove(self, board, playNo ):

        self.opponent = 3 - playNo

        self.tree = Tree()
        state = self.getState()
        state.setBoard(board)
        state.setPlayNo(opponent)

        while True:

            promisingNode = self.selectPromisingNode(self):


    def selectPromisingNode(self, rootNode):

        while rootNode.checkChildArray():
            pass


class UCT(object):

    def __init__(self):
        pass

    def uctValue(self, totalVisit, nodeWinScore, nodeVisit):

        if (nodeVisit == 0):
            return Integer.MAX_VALUE

        return ( nodeWinScore /  nodeVisit)
          + 1.41 * Math.sqrt(Math.log(totalVisit) / nodeVisit)

    def findBestNodeWithUCT(self, node):
        parentVisit = node.getState().getVisitCount()
        winscore = node.getState().getWinScore()
        visitcount = node.getState().getVisitCount()
        score = self.uctValue(parentVisit, winscore, visitcount)

        return np.max( node.getChildArray(), score )
