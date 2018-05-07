import gym


class RandomAgent:
    def act(self, env: gym.Env, s):
        a = env.action_space.sample()
        return a

    def learn(self, s, a, r, ns):
        pass
