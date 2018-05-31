import gym
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from collections import OrderedDict
from src.utility.numpy_util import np_contains


class CliffworldEnv(gym.Env):

    def __init__(self):

        # Set environment types
        self._env_types = OrderedDict({'neutral': 'N',
                                       'cliff': 'C',
                                       'goal': 'G',
                                       'start': 'S'})
        # Set rewards
        self._rewards = OrderedDict({'neutral': -5,
                                     'cliff': -5000,
                                     'goal': -1,
                                     'start': -15})
        self.build_world()

        self._states = [(y, x) for x in range(
            0, self._world.shape[1]) for y in range(0, self._world.shape[0])]

        self._actions = {'>': np.array([0, 1]),
                         'v': np.array([1, 0]),
                         '<': np.array([0, -1]),
                         '^': np.array([-1, 0])}

        self._action_to_str = {0: '>', 1: 'v', 2: '<', 3: '^'}

        self.action_space = gym.spaces.Discrete(4)
        self._state = self._get_start()

        self._states_colors = matplotlib.colors.ListedColormap(
            ['#9A9A9A', '#990000',  '#009900', '#000099'])
        self._cmap_default = 'Blues'
        self._cpal_default = sns.color_palette("Blues_d")
        sns.set_style('white')
        sns.set_context('poster')

        super().__init__()

    def _get_start(self):
        return next(zip(*np.where(self._world == self.get_env_type_code('start'))))

    def get_env_type_code(self, env_type_name):
        return self.get_encoding(self._env_types, env_type_name)

    def get_encoding(self, encodingDict, key_name):
        return list(encodingDict.keys()).index(key_name)

    def render(self, mode='agent', ss=None, Q=None, A=None):
        x_dim = self._world.shape[1]
        y_dim = self._world.shape[0]

        title = 'Cliffworld'
        cmap = self._states_colors
        cbar = True
        color = self._world

        if mode == 'agent':
            annot = np.empty((y_dim, x_dim), dtype=str)
            annot[self._state] = 'O'
        elif mode == 'reward':
            annot = np.empty((y_dim, x_dim), dtype=str)
        elif mode == 'path':
            annot = np.empty((y_dim, x_dim), dtype=str)
            for s in ss:
                if s in self._terminal:
                    annot[s] = 'X'
                else:
                    annot[s] = 'O'
        elif mode == 'policy':
            title = 'Q-function policy'
            cmap = self._cmap_default
            cbar = True
            annot = np.empty((y_dim, x_dim), dtype=str)
            for s in self._states:
                if len(A[s]) == 0:
                    annot[s] = 'u'
                else:
                    qs = {actions: Q[s, actions]
                          for actions in A[s]}
                    action = max(qs, key=qs.get)
                    value = qs[action]
                    annot[s] = self._action_to_str[action]
                color[s] = value
            annot[self._world == self.get_env_type_code('goal')] = 'g'
        plt.subplots(figsize=(10, 10))
        sns.set_style("white")
        sns.set_context("poster")

        ax = sns.heatmap(color, cmap=cmap, annot=annot,
                         cbar=cbar, square=True, linewidths=1, fmt='')

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(np.arange(
            0.5, len(self._env_types)+0.5, 0.7))
        colorbar.set_ticklabels(['{} ({})'.format(key, value)
                                 for key, value in self._rewards.items()])

        ax.set_title(title)
        ax.set(ylabel='y', xlabel='x')

    def step(self, action: int):
        a = self._actions[self._action_to_str[action]]
        self._state = (np.clip(
            self._state[0] + a[0], 0, self._world.shape[0]-1),
            np.clip(self._state[1] + a[1], 0, self._world.shape[1]-1))

        r = self._world_rewards[self._state]
        t = self._state in self._terminal

        return self._state, r, t, {}

    def reset(self):
        self._state = self._get_start()
        return self._state

    def _create_world_rewards(self):
        world_rewards = np.zeros(shape=self._world.shape, dtype=int)
        for key, value in self._rewards.items():
            world_rewards[self._world == self.get_env_type_code(
                key)] = self._rewards[key]

        return world_rewards

    def build_world(self, world_string=12*'N'+'\n'+12*'N'+'\n'+12*'N'+'\n' +
                    'S'+10*'C'+'G'):

        proper_string = self.check_world_string(world_string)

        if not proper_string:
            print('wrong input, try again.')
            return

        self._world = self._create_world(world_string)

        self._terminal = list(zip(*np.where(
            self._world == self.get_env_type_code('goal'))))+list(zip(*np.where(
                self._world == self.get_env_type_code('cliff'))))

        self._world_rewards = self._create_world_rewards()

    def _create_world(self, string):
        world_char_array = np.array([list(line)
                                     for line in string.splitlines()])

        world = np.zeros(shape=world_char_array.shape)

        env_type_indices = {}
        for key, env_type in self._env_types.items():
            env_type_indices[key] = np.where(env_type == world_char_array)
            world[env_type_indices[key]] = self.get_env_type_code(key)

        return world

    def check_world_string(self, string):
        allowed_input = list(self._env_types.values())
        search = re.compile(r'[^'+''.join(allowed_input)+'\n]').search
        return not bool(search(string))

    def set_rewards(self, rewards_dict):
        self._rewards = rewards_dict
