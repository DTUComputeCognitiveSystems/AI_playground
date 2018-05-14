import numpy as np
import random


class ReplayMemory(object):
    def __init__(self, config):
        self.width = int(config['in_width'])
        self.height = int(config['in_height'])
        self.max_size = int(config['hist_size'])
        self.state_frames = int(config['state_frames'])
        self.batch_size = int(config['batch_size'])
        self.states = np.empty([self.max_size, self.width, self.height], dtype=np.uint8)
        self.actions = np.empty([self.max_size], dtype=np.uint8)
        self.rewards = np.empty([self.max_size], dtype=np.uint32)
        self.terminals = np.empty([self.max_size], dtype=np.bool)

        self.count = 0
        self.current = 0

    def get_minibatch(self):
        assert self.count > self.state_frames, "Replay memory contains too few states."

        indices = []
        s = np.empty((self.batch_size, self.width, self.height, self.state_frames))
        ns = np.empty((self.batch_size, self.width, self.height, self.state_frames))
        while len(indices) < self.batch_size:
            while True:
                idx = random.randint(self.state_frames, self.count - 1)
                # if we wrap over current pointer, try again
                if idx >= self.current > idx - self.state_frames:
                    continue
                # if we cross a terminal state, try again (last state can be terminal)
                if self.terminals[idx - 1 - self.state_frames:idx].any():
                    continue
                break

            s[len(indices), ...] = self.get_state(idx - 1)
            ns[len(indices), ...] = self.get_state(idx)
            indices.append(idx)

        a = self.actions[indices]
        r = self.rewards[indices]
        t = self.terminals[indices]

        return s, a, r, ns, t

    def get_state(self, idx):
        assert self.count > 0, "Replay memory is empty."
        assert idx < self.count, "idx not in range"

        if idx > self.state_frames:
            view = self.states[(idx - self.state_frames + 1):idx + 1, ...]
        else:
            indices = [(idx - i) % self.max_size for i in reversed(range(self.state_frames))]
            view = self.states[indices, ...]

        # make the order correct (this shouldn't be expensive since it's just a view and no copying is done)
        view = view.transpose((1, 2, 0))
        view.shape = (1, self.width, self.height, self.state_frames)
        return view

    def get_current(self):
        return self.get_state(self.current - 1)

    def add(self, s, a, r, t):
        # state is after transition
        assert s.shape == (self.width, self.height), "State has wrong dimensions"

        self.states[self.current, ...] = s
        self.actions[self.current] = a
        self.rewards[self.current] = r
        self.terminals[self.current] = t

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.max_size

