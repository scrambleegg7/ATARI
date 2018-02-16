from state import State

import random
import math
import hashlib
import networkx as nx
from ATARI.MCTS import get_module_logger

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR=1/math.sqrt(2.0)

mylogger = get_module_logger(__name__)


class Node():

    def __init__(self, state, parent=None, graph=nx.DiGraph()):
        self.visits=1
        self.reward=0.0
        self.state=state
        self.children=[]
        self.parent=parent
        self.G = graph

    def add_child(self,child_state):

        child_node_last_move = int(child_state.moves[-1])

        if len(self.state.moves) > 0:
            child_node_prev_move = self.state.moves[-1]
            self.G.add_edge(child_node_prev_move, child_node_last_move)
        else:
            self.G.add_node(child_node_last_move)

        child=Node(child_state,parent=self,graph=self.G)
        self.children.append(child)

        #mylogger.warning("add node prev:%s last:%s ", node_prev_move, child_node_last_move)

    def getGraph(self):
        return self.G

    def update(self,reward):
        self.reward+=reward
        self.visits+=1
    def fully_expanded(self):

        lengthOfChildren = len(self.children)
        node_initial_space = self.state.initial_space
        mylogger.debug("# check children fully expanded under Node.. (Node)")
        mylogger.debug("# length of Children %d" % lengthOfChildren)
        mylogger.debug("# initial_space of Node %s" % node_initial_space)

        if lengthOfChildren==node_initial_space:
            mylogger.debug("Sorry, but Node fully expanded ..... (Node)")
            return True

        if self.state.opposit_predict_win:
            mylogger.debug("opposit_predict_win ....")
            return True

        return False
    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f state: %s; parent %s"%(len(self.children),self.visits,self.reward, self.state, self.parent)
        return s
