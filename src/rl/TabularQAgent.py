import gym
from collections import defaultdict
from random import random


class TabularQAgent:
    def __init__(self, alpha: float, epsilon: float, gamma: float):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(int)
        self.A = defaultdict(set)

    def act(self, env: gym.Env, s) -> int:
        if random() > self.epsilon:
            a = self._greedy(env, s)
        else:
            a = env.action_space.sample()
        return a

    def learn(self, s, a, r, ns, t):
        #print('Learning from ', s, ' -> ', a, ' -> ', ns, ', with reward ', r, '.')
        self.Q[tuple(s), a] += self.alpha * self._td(s, a, r, ns)
        self.A[tuple(s)].add(a)

    def _greedy(self, env: gym.Env, s) -> int:
        if len(self.A[tuple(s)]) == 0:
            a = env.action_space.sample()
        else:
            qs = {action: self.Q[tuple(s), action] for action in self.A[tuple(s)]}
            a = max(qs, key=qs.get)
        return a

    def _td(self, s, a, r, ns) -> float:
        if len(self.A[tuple(ns)]) == 0:
            max_q = 0
        else:
            qs = {actions: self.Q[tuple(ns), actions] for actions in self.A[tuple(ns)]}
            max_q_a = max(qs, key=qs.get)
            max_q = qs[max_q_a]
        return r + self.gamma * max_q - self.Q[tuple(s), a]


