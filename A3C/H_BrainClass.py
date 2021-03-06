#

#
#  The Brain class encapsulates the neural network,
#  TensorFlow graph definition and related computation. Let’s look at its parts in order.
#

import numpy as np

import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K


RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 # entropy coefficient

class Brain:

    train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, NUM_STATE, NUM_ACTIONS):

        self.NUM_STATE = NUM_STATE
        self.NUM_ACTIONS = NUM_ACTIONS

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()	# avoid modifications



    def _build_model(self):

        l_input = Input( batch_shape=(None, self.NUM_STATE) )
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(self.NUM_ACTIONS, activation='softmax')(l_dense)
        out_value   = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()	# have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
        loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)	# yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
                return 									# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v
