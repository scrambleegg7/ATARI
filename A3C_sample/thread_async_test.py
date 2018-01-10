import threading
import time
import logging, os

from queue import Queue
from time import sleep, gmtime, strftime

from myObject import myObject

import numpy as np
import tensorflow as tf

from custom_env import CustomGym
from agent import Agent
from a3agent import A3CNet

# FLAGS
T_MAX = 100000
NUM_THREADS = 4
INITIAL_LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
VERBOSE_EVERY = 4000
TESTING = False

I_ASYNC_UPDATE = 5



logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

training_finished = False
lock_queue = threading.Lock()


class Summary:
    def __init__(self, logdir, agent):
        with tf.variable_scope('summary'):

            summarising = ['episode_avg_reward', 'avg_value']
            self.agent = agent
            self.writer = tf.summary.FileWriter(logdir, self.agent.sess.graph)
            self.summary_ops = {}
            self.summary_vars = {}
            self.summary_ph = {}
            for s in summarising:
                self.summary_vars[s] = tf.Variable(0.0)
                self.summary_ph[s] = tf.placeholder('float32', name=s)
                self.summary_ops[s] = tf.summary.scalar(s, self.summary_vars[s])
                self.update_ops = []
            for k in self.summary_vars:
                self.update_ops.append(self.summary_vars[k].assign(self.summary_ph[k]))
                self.summary_op = tf.summary.merge(list(self.summary_ops.values()))

    def write_summary(self, summary, t):
        self.agent.sess.run(self.update_ops, {self.summary_ph[k]: v for k, v in summary.items()})
        summary_to_add = self.agent.sess.run(self.summary_op, {self.summary_vars[k]: v for k, v in summary.items()})
        self.writer.add_summary(summary_to_add, global_step=t)


def estimate_reward(agent, env, episodes=10, max_steps=10000):
    episode_rewards = []
    episode_vals = []
    t = 0
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            t += 1
            episode_vals.append(value)
            episode_reward += reward
            if t > max_steps:
                episode_rewards.append(episode_reward)
                return episode_rewards, episode_vals
        episode_rewards.append(episode_reward)
    return episode_rewards, episode_vals

def evaluator(agent, env, T_queue, sess, summary, saver, save_path,):

    T = T_queue.get()
    T_queue.put(T)
    last_time = time.time()
    last_verbose = T

    while T < T_MAX:
        T = T_queue.get()
        T_queue.put(T)
        if T - last_verbose >= VERBOSE_EVERY:
            print( "T", T )
            current_time = time.time()
            print( "Train steps per second", float(T - last_verbose) / (current_time - last_time) )
            last_time = current_time
            last_verbose = T

            print( "Evaluating agent" )
            episode_rewards, episode_vals = estimate_reward(agent, env, episodes=5)
            avg_ep_r = np.mean(episode_rewards)
            avg_val = np.mean(episode_vals)
            print( "Avg ep reward", avg_ep_r, "Average value", avg_val )

            #summary.write_summary({'episode_avg_reward': avg_ep_r, 'avg_value': avg_val}, T)
            checkpoint_file = saver.save(sess, save_path, global_step=T)
            print( "Saved in", checkpoint_file )


        sleep(1.0)



def async_trainer(agent, env, T_queue, sess, thread_idx, saver, save_path, TESTING=False):

    print( "Training thread", thread_idx)
    t = 0
    T = T_queue.get()
    T_queue.put(T+1)

    last_time = time.time()

    terminal = True
    while T < T_MAX:

        # display T counter every 100 steps
        if T % 1000 == 0:
            logging.debug("async Counter -> %d ",T)

        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            terminal = False
            state = env.reset_1()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # Save the current state
            batch_states.append(state)

            # Choose an action randomly according to the policy
            # probabilities. We do this anyway to prevent us having to compute
            # the baseline value separately.
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal.
            state, reward, terminal, _ = env.step_1(action_idx)


            T = T_queue.get()
            T_queue.put(T+1)

            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value[0])

        target_value = 0
        # If the last state was terminal, just put R = 0. Else we want the
        # estimated value of the last state.
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)
        # Reverse the batch target values, so they are in the correct order
        # again.
        #print("target value", target_value)
        batch_target_values.reverse()

        # Test batch targets
        if TESTING:
            temp_rewards = batch_rewards + [last_R]
            test_batch_target_values = []
            print("temp_rewards:", temp_rewards)
            for j in range(len(batch_rewards)):
                test_batch_target_values.append(discount(temp_rewards[j:], DISCOUNT_FACTOR))
            #if not test_equals(batch_target_values, test_batch_target_values,
            #    1e-5):
            #print( "Assertion failed" )
            print( "lastR:",  last_R )
            print( "batch_rewards:", batch_rewards )
            print( "batch_target_values:", batch_target_values )
            print( "test_batch_target_values:", test_batch_target_values )
            print( "baseline_values:", baseline_values )

        # Compute the estimated value of each state
        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)

        # Apply asynchronous gradient update
        agent.train(np.vstack(batch_states), batch_actions, batch_target_values,
        batch_advantages)

    global training_finished
    training_finished = True


def __async_trainer(agent, env, sess, T_queue, i):

    # preparation for counter

    I_ASYNC_UPDATE = 5
    DISCOUNT_FACTOR = .99

    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    last_verbose = T
    #last_time = time()
    last_target_update = T

    start_time = time.time()

    terminal = True
    while T < 100:

        #logging.debug('thread start -> %d , Counter :%d ',i+1,  T)
        #time.sleep(1) # pretend to do something

        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            terminal = False
            state = env.reset_1()

        while not terminal and len(batch_states) < I_ASYNC_UPDATE:
            # Save the current state
            batch_states.append(state)

            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal.
            state, reward, terminal, _ = env.step_1(action_idx)

            # Update counters
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)


            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value[0])

            T_queue.task_done()

        target_value = 0
        # If the last state was terminal, just put R = 0. Else we want the
        # estimated value of the last state.
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + DISCOUNT_FACTOR * target_value
            batch_target_values.append(target_value)
        # Reverse the batch target values, so they are in the correct order
        # again.
        batch_target_values.reverse()

        # Compute the estimated value of each state
        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)

        # Apply asynchronous gradient update
        #agent.train(np.vstack(batch_states), batch_actions, batch_target_values,batch_advantages)


    #T_queue.task_done()

    #logging.debug('Exited , Counter : %d', T)
    #logging.debug('time consumed on %d : %.5f ', i, time.time() - start_time )

    global training_finished
    training_finished = True

def something_staff(worker,myobject):

    thread_name = threading.current_thread().name
    myobject.increment()

    #time.sleep(2) # pretend to do some work.
    with lock_queue:
        print("locked ...",thread_name,worker)


def do_staff(i, myobject):

    x = range(1,10,1)
    x_t = np.random.permutation(x)[0]
    start_time = time.time()

    thread_name = threading.current_thread().name

    while True:
        worker = T_queue.get()
        #exampleJob(worker)
        logging.debug('Starting,  but sleeping %d , Counter :%d ',   x_t ,  worker)
        #time.sleep(x_t)
        something_staff(worker, myobject)

        logging.debug('Exited from %s',thread_name)
        logging.debug('time consumed  : %.5f ', time.time() - start_time )
        T_queue.task_done()

    global training_finished
    training_finished = True

def proc1(i):

    num_threads = 8
    game_name = 'SpaceInvaders-v0'
    save_path = None
    restore = None


    if save_path is None:
        save_path = 'experiments/' + game_name + '/' + \
        strftime("%d-%m-%Y-%H:%M:%S/model", gmtime())
        print("No save path specified, so saving to", save_path)
    if not os.path.exists(save_path):
        logging.debug("%s doesn't exist, so creating" , save_path)
        os.makedirs(save_path)


    processes = []
    envs = []
    for _ in range(num_threads+1):
        #gym_env = gym.make(game_name)
        print( "Assuming ATARI game and playing pixels" )
        env = CustomGym(game_name)
        envs.append(env)

    # Separate out the evaluation environment
    evaluation_env = envs[0]
    envs = envs[1:]

    with tf.Session() as sess:

        agent = Agent(session=sess,
        action_size=envs[0].action_size, model='mnih',
        optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))

        #agent = A3CNet(session=sess,
        #action_size=envs[0].action_size, model='mnih',
        #optimizer=tf.train.AdamOptimizer(INITIAL_LEARNING_RATE))

        # Create a saver, and only keep 2 checkpoints.
        saver = tf.train.Saver(max_to_keep=2)

        T_queue = Queue()

        # Either restore the parameters or don't.
        #if restore is not None:
        #    saver.restore(sess, save_path + '-' + str(restore))
        #    last_T = restore
        #    print( "T was:", last_T )
        #    T_queue.put(last_T)
        #else:
        sess.run(tf.global_variables_initializer())
        T_queue.put(0)

        summary = Summary(save_path, agent)

        # Create a process for each worker
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], T_queue, sess, i, saver, save_path)))
            #envs[i], sess, i, T_queue, saver, save_path,)))

        # Create a process to evaluate the agent
        processes.append(threading.Thread(target=evaluator, args=(agent, evaluation_env, T_queue, sess, summary, saver, save_path,)))

        # Start all the processes
        for p in processes:
            p.daemon = True
            p.start()

        # Until training is finished
        while not training_finished:
            sleep(0.01)

        # Join the processes, so we get this thread back.
        for p in processes:
            p.join()

def proc2(i):

    global T_queue
    T_queue = Queue()

    myobject = myObject()

    processes = []
    num_threads = 4
    for i in range(num_threads):
        processes.append(threading.Thread(target=do_staff, args=(i,myobject) ) )


    start_time = time.time()

    for p in processes:
        p.setDaemon(True)
        p.start()

    #while not training_finished:
        #logging.debug("till training_finished .. wait 20 s")
    #    time.sleep(.1)

    for q_i in range(15):
        T_queue.put(q_i)

    T_queue.join()

    logging.debug('time consumed on proc2 : %.5f ', time.time() - start_time )


def main():
    i = 0

    # start proc1
    proc1(i)

    # start proc2
    #proc2(i)



if __name__ == "__main__":
    main()
