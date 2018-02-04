#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import itertools as itt
import logging

#
# Solve graph : search most efficient path to find best total reward .....
#
#

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s' )



class QAgent(object):

    def __init__(self, points_list, matrix_size=8, goal= 7):

        self.goal = goal
        self.R = np.ones((matrix_size,matrix_size))
        self.R *= -1
        self.gamma = 0.99

        self.Q_matrix = np.zeros_like(self.R)

        for point in points_list:
            print(point)
            if point[1] == goal:
                self.R[point] = 100
            else:
                self.R[point] = 0

            if point[0] == goal:
                self.R[point[::-1]] = 100
            else:
                # reverse of point
                self.R[point[::-1]]= 0

        # add goal point round trip
        self.R[goal,goal]= 100
        #print(R)

    def getStatus(self):

        current_status = np.random.randint(0,self.Q_matrix.shape[0])
        return current_status

    def pickup_action(self,current_status):

        actions = np.where( self.R[current_status,:] >= 0 )[0]
        action_random_choice = np.random.choice(actions,1)
        #logging.debug("act:%s", actions)

        return action_random_choice

    def getScore(self):

        current_status = self.getStatus()
        random_action = self.pickup_action(current_status)

        max_index = np.where( self.Q_matrix[random_action,] == np.max( self.Q_matrix[random_action,] ) )[1]
        #
        #logging.debug("max_index %s",max_index)
        #
        if len(max_index) > 1:
            max_index = np.random.choice(max_index,1)

        self.Q_matrix[current_status,random_action] = self.R[current_status,random_action] + max_index * self.gamma

        if np.sum(self.Q_matrix) > 0:
            return np.sum( self.Q_matrix / np.max(self.Q_matrix) )
        else:
            return 0



def drawNetwork():

    # map cell to cell, add circular cell to goal point
    points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]

    qAgent = QAgent(points_list=points_list)

    #seq = range(10)
    #combi = list(itt.combinations(seq,2))

    #points_list = np.random.permutation(combi)[:7]

    #G=nx.Graph()
    #G.add_edges_from(points_list)
    #pos = nx.spring_layout(G)
    #nx.draw_networkx_nodes(G,pos)
    #nx.draw_networkx_edges(G,pos)
    #nx.draw_networkx_labels(G,pos)
    #plt.show()

    episodes = 1000
    scores = []
    for i in range(episodes):

        score = qAgent.getScore()
        scores.append(score)



    #logging.debug("Score")
    #logging.debug("%s\n",qAgent.Q_matrix)

    plt.plot(range(len(scores)),scores )
    plt.show()

def main():

    drawNetwork()


if __name__ == "__main__":
    main()
