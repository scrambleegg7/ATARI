#
import numpy as np
from time import time
import tensorflow as tf
import gym, time, random, threading

from H_AgentClass import AgentClass

ENV = "CartPole-v0"
EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000



RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

MIN_BATCH = 32
LEARNING_RATE = 5e-3


# THREAD_DELAY parameter controls a delay between steps and enables
# to have more parallel environments than there are CPUs.

class EnvironmentClass(threading.Thread):

	stop_signal = False

	def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):

		threading.Thread.__init__(self)

		self.render = render
		self.env = gym.make(ENV)
		self.agent = AgentClass(eps_start, eps_end, eps_steps)

	def runEpisode(self):
		s = self.env.reset()

		R = 0
		while True:
			time.sleep(THREAD_DELAY) # yield

			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				s_ = None

			self.agent.train(s, a, r, s_)

			s = s_
			R += r

			if done or self.stop_signal:
				break

		print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True
