import numpy as np
import pandas as pds
import random
from collections import deque
import copy

class Maze(object):
    def __init__(self, size=10, blocks_rate=0.1):
        self.size = size if size > 3 else 10
        self.blocks = int((size ** 2) * blocks_rate)
        self.s_list = []
        self.maze_list = []
        self.e_list = []

    def create_mid_lines(self, k):
        if k == 0: self.maze_list.append(self.s_list)
        elif k == self.size - 1: self.maze_list.append(self.e_list)
        else:
            tmp_list = []
            for l in range(0,self.size):
                if l == 0: tmp_list.extend("#")
                elif l == self.size-1: tmp_list.extend("#")
                else:
                    a = random.randint(-1, 0)
                    tmp_list.extend([a])
            self.maze_list.append(tmp_list)

    def insert_blocks(self, k, s_r, e_r):
        b_y = random.randint(1, self.size-2)
        b_x = random.randint(1, self.size-2)
        if [b_y, b_x] == [1, s_r] or [b_y, b_x] == [self.size - 2, e_r]: k = k-1
        else: self.maze_list[b_y][b_x] = "#"

    def generate_maze(self):
        s_r = random.randint(1, (self.size / 2) - 1)
        for i in range(0, self.size):
            if i == s_r: self.s_list.extend("S")
            else: self.s_list.extend("#")
        start_point = [0, s_r]

        e_r = random.randint((self.size / 2) + 1, self.size - 2)
        for j in range(0, self.size):
            if j == e_r: self.e_list.extend([50])
            else: self.e_list.extend("#")
        goal_point = [self.size - 1, e_r]

        for k in range(0, self.size):
            self.create_mid_lines(k)

        for k in range(self.blocks):
            self.insert_blocks(k, s_r, e_r)

        return self.maze_list, start_point, goal_point
