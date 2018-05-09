import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from random import random, sample


class NeuralQAgent:
    def __init__(self, state_dim: int, num_actions: int, alpha: float, epsilon: float, gamma: float):
        self.model = Sequential()
        self.model.add(Dense(6, input_dim=state_dim, activation='tanh'))
        #self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(num_actions, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=alpha))
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = 32
        self.memory = deque(maxlen=10000)

    def act(self, env: gym.Env, s) -> int:
        if random() > self.epsilon:
            a = self._greedy(s)
        else:
            a = env.action_space.sample()
        return a

    def learn(self, s, a, r, ns, t):
        self._remember(s, a, r, ns, t)
        if len(self.memory) < self.batch_size:
            return

        s = np.empty(shape=(self.batch_size, self.state_dim))
        a = np.empty(shape=(self.batch_size, self.num_actions), dtype=int)
        r = np.empty(shape=(self.batch_size, 1))
        ns = np.empty(shape=(self.batch_size, self.state_dim))
        t = np.empty(shape=(self.batch_size, 1), dtype=bool)
        target_f = np.empty(shape=(self.batch_size, 1))
        for i, (s1, a1, r1, ns1, t1) in enumerate(sample(self.memory, self.batch_size)):
            s[i, :] = s1
            a[i, :] = a1
            r[i, :] = r1
            ns[i, :] = ns1
            t[i, :] = t1
        target = r + self.gamma * np.amax(self.model.predict(ns), axis=1, keepdims=True)
        #print(r.shape, (np.amax(self.model.predict(ns), axis=1)).shape, target.shape)
        target_f = self.model.predict(s)
        for i in range(self.batch_size):
            if t[i]:
                target_f[i, a[i]] = r[i]
            else:
                target_f[i, a[i]] = target[i]
        self.model.fit(s, target_f, epochs=1, verbose=0)

    def _greedy(self, s) -> int:
        s = np.expand_dims(s, 0)
        return np.argmax(self.model.predict(s)[0])

    def _remember(self, s, a, r, ns, t):
        self.memory.append((s, a, r, ns, t))

