import random
import math
import hashlib
import logging
import argparse



"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  Some features
of the example by design are that moves do not commute and early mistakes are more costly.
In particular there are two models of best child that one can use
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR=1/math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

class State():

    NUM_TURNS = 10
    GOAL = 0
    MOVES=[2,-2,3,-3]
    MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
    num_moves=len(MOVES)

    def __init__(self, value=0, moves=[], turn=NUM_TURNS):

    	self.value=value
    	self.turn=turn
    	self.moves=moves

    def next_state(self):
    	nextmove=random.choice([x*self.turn for x  in self.MOVES])
    	next=State(self.value+nextmove, self.moves+[nextmove],self.turn-1)

    	return next

    def terminal(self):
    	if self.turn == 0:
    		return True
    	return False

    def isWin(self):
        return False

    def reward(self):
    	r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
    	return r

    def __hash__(self):
    	return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
    	if hash(self)==hash(other):
    		return True
    	return False

    def __repr__(self):
    	s="Value: %d; Moves: %s"%(self.value,self.moves)
    	return s


def main():

    #current_node = Node(GameState2(board=Board(),turn=0, next_turn=0,test=True))
    current_node=Node(State())
    num_sims = 100

    for l in range(10):

        print("###  level %d  ###", l)

        uctClass = UCTSEARCH(num_sims/(l+1),current_node)
        current_node=uctClass.getCurrent_Node()
        print("Num Children: %d"%len(current_node.children))
        for i,c in enumerate(current_node.children):
        	print(i,c)
        print("Best Child: %s"%current_node.state)

        print("--------------------------------")

if __name__ == "__main__":
    main()
