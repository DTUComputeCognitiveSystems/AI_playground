import gym
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from src.utility.numpy_util import np_contains


class CliffworldEnv(gym.Env):

    def __init__(self):
        self._max_y = 3
        self._max_x = 11
        self._states = [np.array([y, x]) for x in range(0, self._max_x + 1) for y in range(0, self._max_y + 1)]
        self._cliff = [np.array([self._max_y, x]) for x in range(1, self._max_x)]
        self._goal = np.array([self._max_y, self._max_x])
        self._start = np.array([self._max_y, 0])
        self._terminal = self._cliff + [self._goal]
        self._walk_reward = -1
        self._cliff_reward = -100
        self._actions = {'>': np.array([0, 1]),
                         'v': np.array([1, 0]),
                         '<': np.array([0, -1]),
                         '^': np.array([-1, 0])}
        self._action_to_str = {0: '>', 1: 'v', 2: '<', 3: '^'}
        self.action_space = gym.spaces.Discrete(4)
        self._state = np.array(self._start)

        self._states_colors = matplotlib.colors.ListedColormap(['#9A9A9A', '#990000', '#009900', '#000099'])
        self._cmap_default = 'Blues'
        self._cpal_default = sns.color_palette("Blues_d")
        sns.set_style('white')
        sns.set_context('poster')

        super(CliffworldEnv, self).__init__()

    def render(self, mode='agent', ss=None, Q=None, A=None):
        title = 'Cliffworld'
        cmap = self._states_colors
        cbar = False
        color = np.zeros((self._max_y + 1, self._max_x + 1), dtype=float)
        for s in self._states:
            if np.array_equal(s, self._start):
                color[tuple(s)] = 4
            elif np.array_equal(s, self._goal):
                color[tuple(s)] = 2
            elif np_contains(self._cliff, s):
                color[tuple(s)] = 1

        if mode == 'agent':
            annot = np.empty((self._max_y + 1, self._max_x + 1), dtype=str)
            annot[tuple(self._state)] = 'O'
        elif mode == 'reward':
            annot = np.full((self._max_y + 1, self._max_x + 1), self._walk_reward, dtype=int)
            for s in self._states:
                if np_contains(self._cliff, s):
                    annot[tuple(s)] = self._cliff_reward
        elif mode == 'path':
            annot = np.empty((4, 12), dtype=str)
            for s in ss:
                if np_contains(self._terminal, np.array(s)):
                    annot[s] = 'X'
                else:
                    annot[s] = 'O'
        elif mode == 'policy':
            title = 'Q-function policy'
            cmap = self._cmap_default
            cbar = True
            annot = np.empty((self._max_y + 1, self._max_x + 1), dtype=str)
            for s in self._states:
                if len(A[tuple(s)]) == 0:
                    annot[tuple(s)] = 'u'
                else:
                    qs = {actions: Q[tuple(s), actions] for actions in A[tuple(s)]}
                    action = max(qs, key=qs.get)
                    value = qs[action]
                    annot[tuple(s)] = self._action_to_str[action]
                color[tuple(s)] = value
            annot[self._max_y, self._max_x] = 'g'
        plt.subplots(figsize=(10, 10))
        sns.set_style("white")
        sns.set_context("poster")
        ax = sns.heatmap(color, cmap=cmap, annot=annot, cbar=cbar, square=True, linewidths=1, fmt='')
        ax.set_title(title)
        ax.set(ylabel='y', xlabel='x')

    def step(self, action: int):
        a = self._actions[self._action_to_str[action]]
        self._state[0] = np.clip(self._state[0] + a[0], 0, self._max_y)
        self._state[1] = np.clip(self._state[1] + a[1], 0, self._max_x)
        r = self._cliff_reward if np_contains(self._cliff, self._state) else self._walk_reward
        t = np_contains(self._terminal, self._state)
        return self._state, r, t, {}

    def reset(self):
        self._state = np.array(self._start)
        return self._state


