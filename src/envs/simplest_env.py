import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np

class SimplestEnv(Env):
    def __init__(self,
                 benefits_by_state,
                 T,):
        self.logger = logging.getLogger(__name__)
        self.seed()

        self.n = benefits_by_state[0].shape[0]
        self.m = benefits_by_state[0].shape[1]
        self.num_states = len(benefits_by_state)

        self.benefits_by_state = benefits_by_state

        self.T = T

        self.curr_state = 0
        self.curr_step = 0

        #Action space is one action for each task
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.m)] * self.n))

        #Observation space is just what state the game is in
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.num_states)] * self.n))

    def step(self, actions):
        """ Returns reward, terminated, info """
        num_times_tasks_completed = np.zeros(self.m)
        for i in range(self.n):
            num_times_tasks_completed[actions[i]] += 1

        rewards = []
        for i in range(self.n):
            chosen_task = actions[i]
            rewards.append(self.benefits_by_state[self.curr_state][i, chosen_task] / num_times_tasks_completed[chosen_task])

        self.curr_state = actions[0] % self.num_states
        self.curr_step += 1
        
        nobs = [self.curr_state for _ in range(self.n)]

        done = self.curr_step >= self.T
        dones = [done for _ in range(self.n)]
        return nobs, rewards, dones, {}

    def reset(self):
        """ Resets environment to state 1."""
        self.curr_state = 0
        self.curr_step = 0

        return [self.curr_state for _ in range(self.n)]

    def close(self):
        return True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]