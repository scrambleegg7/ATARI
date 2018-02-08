#
import numpy as np
import random

from node2 import Node

from Board import Board
from GameState2 import GameState2

from state import State


from logging import getLogger, StreamHandler, DEBUG
from logging import Formatter
import logging
from logClass import MyHandler

import networkx as nx

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.propagate = False

SCALAR = 1. / np.sqrt(2)


class UCTSEARCH(object):

    def __init__(self,budget, root):
        self.log = getLogger("UCTSEARCH")
        self.current_node = self.process(budget, root)

    def __del__(self):
        root = self.log
        map(root.removeHandler, root.handlers[:])
        map(root.removeFilter, root.filters[:])

    def getCurrent_Node(self):
        return self.current_node

    def process(self, budget, root):

        self.log.warning("** UCTSEARCH budget. (UCTSEARCH Class) : %d", budget)
        for i in range(int(budget)):
            self.log.warning("**** budget range. (UCTSEARCH Class) : [ %d ]" % i )
            front  = self.treePolicy(root)
            reward = self.defaultPolicy(front.state)
            self.backup(front,reward)

        return self.BESTCHILD(root,SCALAR)

    def treePolicy(self,node):

        # node --> root node

        while node.state.terminal()==False:
            if len(node.children)==0:
                return self.EXPAND(node)
            elif random.uniform(0,1)<.5:

                print("pickup BEST CHILD with probability less than50%.." )
                node=self.BESTCHILD(node,SCALAR)

            else:
                if node.fully_expanded()==False:
                    print("Check fully expanded..")
                    return self.EXPAND(node)

                else:
                    print("pickup BEST CHILD with probability than 50%.." )
                    node=self.BESTCHILD(node,SCALAR)

        return node

    def treePolicy_winningPatternCheck(self,node):

        # node --> root node

        while node.state.terminal()==False:
            if len(node.children)==0:
                return self.EXPAND(node)
            elif random.uniform(0,1)<.5:

                print("pickup BEST CHILD with probability less than50%.." )
                node=self.BESTCHILD(node,SCALAR)

            else:
                if node.fully_expanded()==False:
                    print("Check fully expanded..")
                    return self.EXPAND(node)

                else:
                    print("pickup BEST CHILD with probability than 50%.." )
                    node=self.BESTCHILD(node,SCALAR)

        return node


    def defaultPolicy(self,state):

        loop_cnt=1
        while state.isWin() == False:
            if state.terminal():
                break
            state=state.next_state()

            print("DEFAULTPOLICY state..",state)

            loop_cnt += 1

        r = state.reward()
        print("**  reward  ** ", r)

        return r

    def EXPAND(self, node):

        print("EXPAND .....(add child on new node.)")
        node_child_length = len(node.children)
        self.log.warning("node children length (EXPAND) %d", node_child_length)

        tried_children=[c.state for c in node.children]
        new_state=node.state.next_state()
        while new_state in tried_children:

            new_state=node.state.next_state()

        print(new_state)

        node.add_child(new_state)
        return node.children[-1]

    def BESTCHILD(self, node, scalar):

        bestscore=-999.
        bestchildren=[]
        for idx, c in enumerate( node.children):
            exploit=c.reward/c.visits
            explore=np.sqrt(2.0 * np.log(node.visits)/float(c.visits))


            score=exploit+scalar*explore

            print("%d Child score (BESTCHILD) : %.4f" % (idx,score)  )
            print("   node visits %d  child visits %d" % ( node.visits, c.visits) )
            print("   exploit %.4f  explore %.4f" % (exploit, explore))
            print("   child move history.", c.state.moves)
            print("   child next turn.", c.state.next_turn)

            if score==bestscore:
                bestchildren.append(c)

            if score>bestscore:

                bestchildren=[c]
                bestscore=score

        if len(bestchildren) == 0:
            print("## sorry, NO CHILDREN  ERROR ! ##")


        return random.choice(bestchildren)

    def backup(self,node,reward):
        while node!=None:
            node.visits+=1
            node.reward+=reward
            node=node.parent
        return


def StateStart():

    current_node=Node(State())
    uctClass=UCTSEARCH(5,current_node)
    current_node = uctClass.getCurrent_Node()

    print("Num Children: %d"%len(current_node.children))
    for i,c in enumerate(current_node.children):
        print(i,c)
    print("Best Child: %s"%current_node.state)


def GameStart():

    current_node = Node(GameState2(moves=[], board=Board(),turn=0, next_turn=0,test=True))
    uctClass = UCTSEARCH(10,current_node)
    current_node = uctClass.getCurrent_Node()

    print("Num Children: %d"%len(current_node.children))
    for i,c in enumerate(current_node.children):
    	print(i,c)
    print("Best Child: %s"%current_node.state)
    print("--------------------------------")

def GameStartS0():

    # Start Game with S0 1 3 5 7
    board_array = np.array( [" "] * 10  )
    board_array[2] = "O"
    board_array[3] = "X"
    board_array[7] = "X"
    board_array[5] = "O"

    myBoard = Board(board=board_array)

    current_node = Node(GameState2(moves=[3, 2, 7, 5], board=myBoard,turn=4, next_turn="a"))
    uctClass = UCTSEARCH(100,current_node)
    current_node = uctClass.getCurrent_Node()

    print("Num Children: %d"%len(current_node.children))
    for i,c in enumerate(current_node.children):
    	print(i,c)
    print("Best Child: %s"%current_node.state)
    print("--------------------------------")



def main():

    GameStartS0()
    #StateStart()

if __name__ == "__main__":
    main()
