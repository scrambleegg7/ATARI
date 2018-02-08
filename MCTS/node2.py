from state import State

import random
import math
import hashlib
import logging
import argparse

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR=1/math.sqrt(2.0)



class Node():

    def __init__(self, state, parent=None):
        self.visits=1
        self.reward=0.0
        self.state=state
        self.children=[]
        self.parent=parent

    def add_child(self,child_state):
        child=Node(child_state,self)
        self.children.append(child)
    def update(self,reward):
        self.reward+=reward
        self.visits+=1
    def fully_expanded(self):

        lengthOfChildren = len(self.children)
        node_initial_space = self.state.initial_space
        print("# check children fully expanded under Node.. (Node)")
        print("# length of Children %d" % lengthOfChildren)
        print("# initial_space of Node %s" % node_initial_space)

        if lengthOfChildren==node_initial_space:
            print("Sorry, but Node fully expanded ..... (Node)")
            return True

        if self.state.opposit_predict_win:
            print("opposit_predict_win ....")
            return True

        return False
    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f state: %s; parent %s"%(len(self.children),self.visits,self.reward, self.state, self.parent)
        return s


def UCTSEARCH(budget,root):
    for iter in range(int(budget)):
        if iter%10000==9999:
            logger.info("simulation: %d"%iter)
            logger.info(root)
        front=TREEPOLICY(root)
        reward=DEFAULTPOLICY(front.state)
        BACKUP(front,reward)
    return BESTCHILD(root,0)

def TREEPOLICY(node):
    #a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal()==False:
        if len(node.children)==0:
            return EXPAND(node)
        elif random.uniform(0,1)<.5:
            print("pickup BEST CHILD with probability less than50%.." )
            node=BESTCHILD(node,SCALAR)
        else:
            if node.fully_expanded()==False:
                print("Check fully expanded..")
                return EXPAND(node)
            else:
                print("pickup BEST CHILD with probability than 50%.." )
                node=BESTCHILD(node,SCALAR)
    return node

def EXPAND(node):
    tried_children=[c.state for c in node.children]
    new_state=node.state.next_state()
    while new_state in tried_children:
        new_state=node.state.next_state()
    print("** EXPAND **")
    print(new_state)
    node.add_child(new_state)
    return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
    bestscore=-9999
    bestchildren=[]
    for idx, c in enumerate(node.children):
        exploit=c.reward/c.visits
        explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))
        score=exploit+scalar*explore
        logger.warning("%d Child score (BESTCHILD) : %.4f" % (idx,score)  )
        if score==bestscore:
            bestchildren.append(c)
        if score>bestscore:
            bestchildren=[c]
            bestscore=score
    if len(bestchildren)==0:
        logger.warning("OOPS: no best child found, probably fatal")
    return random.choice(bestchildren)

def DEFAULTPOLICY(state):
    while state.isWin()==False:
        if state.terminal():
            break
        state=state.next_state()
        print("DEFAULTPOLICY state..",state)
    return state.reward()

def BACKUP(node,reward):
    while node!=None:
        node.visits+=1
        node.reward+=reward
        print("BACKUP - node printing..",node)

        node=node.parent
    return
