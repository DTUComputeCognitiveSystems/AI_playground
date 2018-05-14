import random
import numpy as np
import gym
import cv2
from random import random
from src.rl.ReplayMemory import ReplayMemory


class AtariAgent(object):
    def __init__(self, env: gym.Env, net, config):
        self.mem = ReplayMemory(config)
        self.env = env
        self.net = net

        self.eps = config['eps']
        self.max_reward = -np.inf
        self.buf_size = 4

        self.state_buf = np.zeros(shape=(1, 84, 84, self.buf_size), dtype=int)

    def act(self, env: gym.Env, s) -> int:
        s = self._scale(s)
        self.state_buf = np.roll(self.state_buf, shift=-1, axis=3)
        self.state_buf[0, :, :, -1] = s

        if random() > self.eps:
            a = self.net.predict(self.state_buf)
        else:
            a = env.action_space.sample()
        return a

    def learn(self, s, a, r, ns, t):
        s = self._scale(s)
        self.mem.add(s, a, r, t)
        if self.mem.count < self.mem.batch_size:
            return

        s, a, r, ns, t = self.mem.get_minibatch()
        self.net.train(s, self._onehot_actions(a), r, ns, t)

    def sync_target(self):
        self.net.sync_target()

    def reset(self):
        self.state_buf = np.zeros(shape=(1, 84, 84, self.buf_size), dtype=int)

    def _scale(self, s):
        s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        return cv2.resize(s, (84, 84))

    def _onehot_actions(self, actions):
        size = len(actions)
        onehot = np.zeros((size, self.env.action_space.n))
        for i in range(size):
            onehot[i, actions[i]] = 1
        return onehot