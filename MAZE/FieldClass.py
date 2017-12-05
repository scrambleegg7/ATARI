import numpy as np
import pandas as pds
import random
from collections import deque
import copy


class Field(object):
    def __init__(self, maze, start_point, goal_point):
        self.maze = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1,0],[-1,0],[0,1],[0,-1]]

    def display(self, point=None):
        field_data = copy.deepcopy(self.maze)
        if not point is None:
                y, x = point
                field_data[y][x] = "@@"
        else:
                point = ""
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):
        movables = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 < x < len(self.maze) and
                       0 <= y <= len(self.maze) - 1 and
                       maze[y][x] != "#" and
                       maze[y][x] != "S"):
                    continue
                movables.append([y,x])
            if len(movables) != 0:
                return movables
            else:
                return None

    def get_val(self, state):
        y, x = state
        if state == self.start_point: return 0, False
        else:
            v = float(self.maze[y][x])
            if state == self.goal_point:
                return v, True
            else:
                return v, False
