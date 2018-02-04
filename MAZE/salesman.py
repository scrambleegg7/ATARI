#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as itt
import networkx as nx

import logging


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def makeRoute():

    x = range(10)
    perm_x2 = list(itt.permutations(x,2))
    points_list = np.array(perm_x2)

    G=nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()

    route_list = np.array([
        [   0,    22,    16,    21,    19,    24,    17,    21,    30,    29],
        [    16,   0,    16,    29,    16,    20,    20,    24,    26,    19],
        [    19,    27,   0,    22,    18,    20,    30,    26,    21,    27],
        [    28,    29,    24,   0,    28,    26,    18,    17,    16,    21],
        [    18,    26,    24,    21,   0,    26,    20,    19,    24,    20],
        [    16,    22,    26,    25,    26,   0,    26,    30,    28,    27],
        [    17,    20,    18,    20,    30,    28,   0,    30,    29,    16],
        [    24,    19,    16,    20,    19,    30,    23,   0,    22,    22],
        [    26,    29,    18,    22,    21,    20,    30,    22,   0,    17],
        [    19,    28,    29,    18,    23,    23,    30,    28,    21,   0] ] )


    logging.debug(route_list.shape)
    route_list = 100 - route_list
    logging.debug("new route_list %s\n",route_list)


def main():

    makeRoute()

if __name__ == "__main__":
    main()
