import gym
import numpy as np


def run_episode(env: gym.Env, agent, learn=False, max_length=1000) -> float:
    s = np.array(env.reset())
    n = 0
    sum_r = 0
    while n < max_length:
        a = agent.act(env, s)    #: Take the current state as input and compute an action.
        ns, r, t, _ = env.step(a)   #: Take the action and compute the changed state.
        if learn:
            agent.learn(s, a, r, ns)                #: Learn.
        s = np.array(ns)                            #: Newstate becomes the current state for next iteration.
        n += 1
        sum_r += r
        if t:
            break
    return sum_r


# same as run_episode, but returns a visited state set 'ss'
def run_episode_ss(env: gym.Env, agent, learn=False, max_length=1000) -> (float, set):
    ss = set()
    s = np.array(env.reset())
    ss.add(tuple(s))
    n = 0
    sum_r = 0
    while n < max_length:
        a = agent.act(env, s)    #: Take the current state as input and compute an action.
        ns, r, t, _ = env.step(a)   #: Take the action and compute the changed state.
        if learn:
            agent.learn(s, a, r, ns)                #: Learn.
        s = np.array(ns)                            #: Newstate becomes the current state for next iteration.
        ss.add(tuple(s))
        n += 1
        sum_r += r
        if t:
            break
    return sum_r, ss
