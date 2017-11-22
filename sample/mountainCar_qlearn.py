import gym
import numpy as np

env = gym.make('MountainCar-v0')

goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 5000
last_time_steps = np.zeros(num_consecutive_iterations)

q_table = np.random.uniform(low=-100, high=-1, size=(100 ** 2, env.action_space.n))

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    # change status to 6 x 6 indices = [0 35]
    # then search matrix 256 x 2 actions..
    car_velocity, car_pos = observation
    digitized = [np.digitize(car_pos, bins=bins(-1.2, 0.6 , 100)),
                 np.digitize(car_velocity, bins=bins(-0.07, 0.07, 100))]
    # [0  255]
    #return sum([x * (100 ** i) for i, x in enumerate(digitized)])
    return digitized

def get_action(state, action, observation, reward):
    next_state = digitize_state(observation)

    epsilon = 0.1
    r = np.random.uniform(0,1)
    #
    # greedy aproach
    #
    if r < epsilon:
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0,1])

    # update table ....
    alpha = 0.2
    gamma = 0.99
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action])

    return next_action, next_state

for episode in range(num_episodes):
    # init
    observation = env.reset()

    state = digitize_state(observation)
    action = np.argmax(q_table[state])

    episode_reward = 0
    for t in range(max_number_of_steps):
        #env.render()

        # get observation , reward
        observation, reward, done, info = env.step(action)

        #if done:
        #    reward =- 200
        # behavior
        action, state = get_action(state, action, observation, reward)
        episode_reward += reward

        if done:
            #print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1,
            #    last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            break

    if (last_time_steps.mean() >= goal_average_steps): # success if average 195 points over last 100 episodes...
        print('Episode %d train agent successfuly!' % episode)
        break

print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1,
    last_time_steps.mean()))
