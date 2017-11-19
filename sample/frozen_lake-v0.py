#


import numpy as np
import gym
import matplotlib.pyplot as plt

#  This says that the Q-value for a given state (s)
#  and action (a) should represent the current reward (r)
#  plus the maximum discounted (γ) future reward expected
#  according to our own table for the next state (s’)
#  we would end up in.
#  The discount variable allows us to decide
#  how important the possible future rewards are compared to the present reward.

env = gym.make('FrozenLake-v0')

observation = env.reset()

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
print("Q shape", Q.shape)
# Set learning parameters
lr = .8
y = .95 # discount rate
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    # reset state to 0....

    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        # loss
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes) )
print()
print("final Q:")
print( Q )
