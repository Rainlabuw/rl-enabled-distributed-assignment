import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import scipy.optimize

class BenefitObsEnv(Env):
    def __init__(self,
                 benefits_by_state,
                 T,
                 bids_as_actions=False,
                 seed=None,
                 key=None):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        self.n = benefits_by_state[0].shape[0]
        self.m = benefits_by_state[0].shape[1]
        self.num_states = len(benefits_by_state)

        self.benefits_by_state = benefits_by_state

        self.T = T

        self.curr_state = 0
        self.curr_state_onehot = np.zeros(self.num_states)
        self.curr_state_onehot[self.curr_state] = 1

        self.curr_step = 0

        self.bids_as_actions = bids_as_actions
        if self.bids_as_actions:
            #Action space is a bid for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=0, high=np.inf, shape=(self.m,))] * self.n))
        else:
            #Action space is one action for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.m)] * self.n))

        #Observation space is just what state the game is in + the benefits for the agent
        self.obs_space_size = self.num_states + self.m 
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,))] * self.n))

    def step(self, actions):
        """ 
        Input is actions in Discrete form.
        Returns reward, terminated, info. 
        """
        if self.bids_as_actions:
            _, assignments = scipy.optimize.linear_sum_assignment(actions, maximize=True)
        else:
            assignments = [int(a) for a in actions]

        num_times_tasks_completed = np.zeros(self.m)
        for i in range(self.n):
            num_times_tasks_completed[assignments[i]] += 1

        rewards = []
        for i in range(self.n):
            chosen_task = assignments[i]
            rewards.append(self.benefits_by_state[self.curr_state][i, chosen_task] / num_times_tasks_completed[chosen_task])

        #Transition into the state according to the first agents assignment.
        self.curr_state = assignments[0] % self.num_states
        self.curr_state_onehot = np.zeros(self.num_states)
        self.curr_state_onehot[self.curr_state] = 1

        self.curr_step += 1        

        self._obs = [np.concatenate([self.curr_state_onehot, self.benefits_by_state[self.curr_state][i,:]]) for i in range(self.n)]

        done = self.curr_step >= self.T

        return rewards, done, {}

    def reset(self):
        """ Resets environment to state 0."""
        self.curr_state = 0
        self.curr_state_onehot = np.zeros(self.num_states)
        self.curr_state_onehot[self.curr_state] = 1

        self.curr_step = 0

        self._obs = [np.concatenate([self.curr_state_onehot, self.benefits_by_state[self.curr_state][i,:]]) for i in range(self.n)]

        return self.get_obs(), self.get_state()

    def close(self):
        return True

    def seed(self, seed=None):
        if seed is None:
            self.np_random, self._seed = seeding.np_random(seed)
        else:
            self._seed = seed

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_space_size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n * self.get_obs_size()

    def get_avail_actions(self):
        """ All actions are available."""
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (all actions are always available)"""
        return [1] * self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.m
    
    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "m": self.get_total_actions(),
                    "n": self.n,
                    "T": self.T}
        return env_info
