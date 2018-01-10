#

import numpy as np  
import pandas as pd   
import matplotlib.pyplot as plt   

from ACAgentClass import ACAgentClass


class WorkerClass(object):


    def __init__(self,game,name,s_size,a_size,trainer,saver,model_path):
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = ACAgentClass(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name) 
    
    def setModel(self):
        pass