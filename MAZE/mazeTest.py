#
import pandas as pd
import numpy as np

from FieldClass import Field
from MazeClass import Maze

from ATARI.DQN.AgentClass_v2 import AgentClass

def choose_best_action(state, movables):
    best_actions = []
    max_act_value = -100
    for a in movables:
        np_action = np.array([[state, a]])
        print("np_action")
        print(np_action)
            #act_value = self.model.predict(np_action)
            #if act_value > max_act_value:
            #    best_actions = [a,]
            #    max_act_value = act_value
            #elif act_value == max_act_value:
            #    best_actions.append(a)
        #return random.choice(best_actions)


size = 10
barriar_rate = 0.1

maze_1 = Maze(size, barriar_rate)
maze, start_point, goal_point = maze_1.generate_maze()
maze_field = Field(maze, start_point, goal_point)

maze_field.display()
print("start end point")
print(start_point, goal_point)

action = maze_field.get_actions(state=start_point)
print("action")
print(action)

#myAgent = AgentClass(2,3)
state = np.array(start_point)

#print(state)
print("state, action 0")
print(np.array( [[state,action[0]]] ) )

choose_best_action(state,action)
