#
import numpy as np
import random
import matplotlib.pyplot as plt

from node2 import Node
from Board import Board
from GameState2 import GameState2
import networkx as nx

from ATARI.MCTS import get_module_logger

SCALAR = 1. / np.sqrt(2)


mylogger = get_module_logger(__name__)


class UCTSEARCH(object):

    def __init__(self,budget, root):
        self.current_node = self.process(budget, root)

    #def __del__(self):
        #root = mylogger
        #map(root.removeHandler, root.handlers[:])
        #map(root.removeFilter, root.filters[:])

    def getCurrent_Node(self):
        return self.current_node

    def process(self, budget, root):

        mylogger.debug("** UCTSEARCH budget. (UCTSEARCH Class) : %d", budget)
        for i in range(int(budget)):
            mylogger.debug("**** budget range. (UCTSEARCH Class) : [ %d ]" % i )
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

                mylogger.debug("pickup BEST CHILD with probability less than50%.." )
                node=self.BESTCHILD(node,SCALAR)

            else:
                if node.fully_expanded()==False:
                    mylogger.debug("Check fully expanded..")
                    return self.EXPAND(node)

                else:
                    mylogger.debug("pickup BEST CHILD with probability than 50%.." )
                    node=self.BESTCHILD(node,SCALAR)

        return node

    def treePolicy_winningPatternCheck(self,node):

        # node --> root node

        while node.state.terminal()==False:
            if len(node.children)==0:
                return self.EXPAND(node)
            elif random.uniform(0,1)<.5:

                mylogger.debug("pickup BEST CHILD with probability less than50%.." )
                node=self.BESTCHILD(node,SCALAR)

            else:
                if node.fully_expanded()==False:
                    mylogger.debug("Check fully expanded..")
                    return self.EXPAND(node)

                else:
                    mylogger.debug("pickup BEST CHILD with probability than 50%.." )
                    node=self.BESTCHILD(node,SCALAR)

        return node


    def defaultPolicy(self,state):

        loop_cnt=1
        while state.isWin() == False:
            if state.terminal():
                break
            state=state.next_state()

            mylogger.debug("DEFAULTPOLICY state..%s ",state)

            loop_cnt += 1

        r = state.reward()
        mylogger.debug("**  reward  ** %.1f ", r)

        return r

    def EXPAND(self, node):

        mylogger.debug("EXPAND .....(add child on new node.)")
        node_child_length = len(node.children)
        mylogger.debug("node children length (EXPAND) %d", node_child_length)
        mylogger.debug("recoreded move of Node %s", node.state.moves  )

        tried_children=[c.state for c in node.children]
        new_state=node.state.next_state()
        while new_state in tried_children:

            new_state=node.state.next_state()

        mylogger.debug(new_state)

        node.add_child(new_state)
        return node.children[-1]

    def BESTCHILD(self, node, scalar):

        bestscore=-999.
        bestchildren=[]
        for idx, c in enumerate( node.children):
            exploit=c.reward/c.visits
            explore=np.sqrt(2.0 * np.log(node.visits)/float(c.visits))


            score=exploit+scalar*explore

            mylogger.debug("%d Child score (BESTCHILD) : %.4f" , idx,score  )
            mylogger.debug("   node visits %d  child visits %d" , node.visits, c.visits )
            mylogger.debug("   exploit %.4f  explore %.4f" , exploit, explore)
            mylogger.debug("   child move history. %s", c.state.moves)
            mylogger.debug("   child next turn. %s", c.state.next_turn)

            if score==bestscore:
                bestchildren.append(c)

            if score>bestscore:

                bestchildren=[c]
                bestscore=score

        if len(bestchildren) == 0:
            mylogger.debug("## sorry, NO CHILDREN  ERROR ! ##")

        return random.choice(bestchildren)

    def backup(self,node,reward):
        while node!=None:
            node.visits+=1
            node.reward+=reward
            node=node.parent
        return


def GameStart():

    current_node = Node(GameState2(moves=[], board=Board(),turn=0, next_turn="a"))
    uctClass = UCTSEARCH(5,current_node)
    current_node = uctClass.getCurrent_Node()

    mylogger.debug("Num Children: %d" , len(current_node.children))
    for i,c in enumerate(current_node.children):
    	mylogger.debug(i,c)
    mylogger.debug("Best Child: %s" , current_node.state)
    mylogger.debug("--------------------------------")

    G = current_node.getGraph()
    nx.draw_networkx(G)
    plt.show()


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

    #logger = get_module_logger(__name__)

    mylogger.debug("Num Children: %d" , len(current_node.children))
    for i,c in enumerate(current_node.children):
        mylogger.debug("%d, %s", i,c)
    mylogger.debug("Best Child: %s" , current_node.state)
    mylogger.debug("--------------------------------")

    G = current_node.getGraph()
    nx.draw_networkx(G)
    plt.show()

def main():

    #GameStartS0()
    GameStart()

if __name__ == "__main__":
    main()
