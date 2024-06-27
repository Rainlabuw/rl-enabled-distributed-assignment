import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
from components.transforms import OneHot
import numpy as np
import torch as th

class DictatorEnv(Env):
    def __init__(self,
                 T,
                 benefits_by_state=None,
                 seed=None,
                 bids_as_actions=False,):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        if benefits_by_state is None:
            self.benefits_by_state = []
            self.benefits_by_state.append(np.array([[2, 3, 0], 
                                                    [0, 2, 3],
                                                    [3, 0, 2]]))
            self.benefits_by_state.append(np.array([[0, 3, 0], 
                                                    [0, 0, 0.1],
                                                    [0.1, 0, 0]]))
            self.benefits_by_state.append(np.array([[0, 0, 3],
                                                    [0.1, 0, 0],
                                                    [0, 0.1, 0]]))

        self.n = self.benefits_by_state[0].shape[0]
        self.m = self.benefits_by_state[0].shape[1]
        self.num_states = len(self.benefits_by_state)

        self.T = T

        self.curr_state = 0
        self.curr_state_onehot = np.zeros(self.num_states)
        self.curr_state_onehot[self.curr_state] = 1
        self.curr_step = 0

        self.bids_as_actions = bids_as_actions

        #Action space is one action for each task
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.m)] * self.n))

        #Observation space is just what state the game is in
        self.obs_space_size = self.get_obs_size()
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.num_states)] * self.n))

        self.scheme, self.preprocess = self._get_scheme_and_preprocess()

    def _get_scheme_and_preprocess(self):
        """
        Contains the scheme and preprocess functions for the environment.

        If a scheme entry contains a "part_of_state" key, the corresponding data will be passed to the agents as part of the state.
        """
        scheme = { #half precision
            "obs": {"vshape": self.obs_space_size, "group": "agents", "dtype": th.float32},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.int64},
            "avail_actions": {
                "vshape": (self.m,),
                "group": "agents",
                "dtype": th.bool,
            },
            "rewards": {"vshape": (self.n,), "dtype": th.float32},
            "terminated": {"vshape": (1,), "dtype": th.bool},
            "beta": {"vshape": (self.n, self.m), "dtype": th.float32, "part_of_state": True},
        }
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.m)])}

        if self.bids_as_actions:
            scheme["actions"] = {"vshape": (self.m,), "group": "agents", "dtype": th.float32}
            preprocess = {}
        
        return scheme, preprocess

    def reset(self):
        """ Resets environment to state 1."""
        self.curr_state = 0
        self.curr_step = 0
        self.curr_state_onehot = np.zeros(self.num_states)
        self.curr_state_onehot[self.curr_state] = 1

        self.beta = self.benefits_by_state[self.curr_state]

        self._obs = [self.curr_state_onehot for _ in range(self.n)]

        return [self.curr_state_onehot for _ in range(self.n)]

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        num_times_tasks_completed = np.zeros(self.m)
        for i in range(self.n):
            num_times_tasks_completed[actions[i]] += 1

        rewards = []
        for i in range(self.n):
            chosen_task = actions[i]
            rewards.append(self.benefits_by_state[self.curr_state][i, chosen_task] / num_times_tasks_completed[chosen_task])

        self.curr_state = actions[0] % self.num_states
        self.curr_state_onehot = np.zeros(self.num_states)
        self.curr_state_onehot[self.curr_state] = 1
        self.curr_step += 1
        
        self.beta = self.benefits_by_state[self.curr_state]

        self._obs = [self.curr_state_onehot for _ in range(self.n)]

        done = self.curr_step >= self.T

        return rewards, done, {}
    
    def get_pretransition_data(self):
        """
        Creates a dict with the data that should be recorded before the transition.

        All "add_pretransition_data()" functions must at least have entries for "beta" and "obs".
        """
        pretransition_data = {
            "obs": [self._obs],
            "avail_actions": [self.get_avail_actions()],
            "beta": [self.beta],
        }
        return pretransition_data
    
    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs
    
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.m
    
    def get_avail_actions(self):
        """ All actions are available."""
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (all actions are always available)"""
        return [1] * self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.m

    def close(self):
        return True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_stats(self):
        return {}
    
    def beta_hat(self, beta):
        """
        Given the info contained in the state, 
        returns a matrix of benefits for each agent-task pair.
        """
        return beta