import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import scipy.optimize
import torch as th
from components.transforms import OneHot

class MockConstellationEnv(Env):
    """
    Model a constellation, where benefits are provided to you for task in several states
    """
    def __init__(self,
                 n, m, T, 
                 L, lambda_,
                 bids_as_actions=False,
                 seed=None,
                 sat_prox_mat=None,#if you want to train on only one specific benefit matrix
                 T_trans=None,
                ):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        self.n = n
        self.m = m
        self.T = T

        if sat_prox_mat is None:
            self.constant_benefits = False
            self.sat_prox_mat = generate_benefits_over_time(n, m, T, 5, 8)
        else:
            self.constant_benefits = True
            self.sat_prox_mat = sat_prox_mat

        #default to all transitions (except nontransitions) being penalized
        self.T_trans = T_trans if T_trans is not None else np.ones((m,m)) - np.eye(m)

        #Number of steps to look into the future
        self.L = L
        self.lambda_ = lambda_
        self.graphs = None #both hardcoded to None for now
        self.init_assignment = None

        #State is the previous assignment
        self.curr_assignment = np.zeros((self.n, self.m))

        self.k = 0

        self.bids_as_actions = bids_as_actions
        if self.bids_as_actions:
            #Action space is a bid for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=0, high=np.inf, shape=(self.m,))] * self.n))
        else:
            #Action space is one action for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.m)] * self.n))

        #Observation space is just the benefits per task for the next L time steps, plus the agent's previous assignment.
        self.obs_space_size = self.L * self.m + self.m
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,))] * self.n))

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
            "prev_assigns": {"vshape": (self.n,), "dtype": th.int64, "part_of_state": True},
            "beta": {"vshape": (self.n, self.m), "dtype": th.float32, "part_of_state": True},
        }
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.m)])}

        if self.bids_as_actions:
            scheme["actions"] = {"vshape": (self.m,), "group": "agents", "dtype": th.float32}
            preprocess = {}
        
        return scheme, preprocess

    def reset(self):
        """ Resets environment."""
        self.curr_assignment = np.zeros((self.n, self.m))
        self.k = 0

        if not self.constant_benefits:
            self.sat_prox_mat = generate_benefits_over_time(self.n, self.m, self.T, 3, 6)
        else:
            pass #if benefits are constant, do nothing because you should use the same benefit matrix

        self.beta = self.sat_prox_mat[:,:,self.k]
        self.prev_assigns = np.random.choice(self.m, self.n, replace=False)

        self._obs = [self.curr_assignment[i,:] for i in range(self.n)]
        for l in range(self.L):
            if self.k + l < self.T:
                self._obs = [np.concatenate([self._obs[i], self.sat_prox_mat[i,:,self.k+l]]) for i in range(self.n)]
            else: #if the episode has ended within your lookahead of L, just pad with zeros
                self._obs = [np.concatenate([self._obs[i], np.zeros(self.m)]) for i in range(self.n)]

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ 
        Input is actions in Discrete form.
        Returns reward, terminated, info.
        """
        if self.bids_as_actions:
            _, assignments = scipy.optimize.linear_sum_assignment(actions, maximize=True)
        else:
            assignments = np.array(actions, dtype=int)

        beta_hat = self.beta_hat(self.beta, self.prev_assigns)

        num_times_tasks_completed = np.zeros(self.m)
        for i in range(self.n):
            num_times_tasks_completed[assignments[i]] += 1

        rewards = []
        for i in range(self.n):
            chosen_task = assignments[i]
            if beta_hat[i, chosen_task] > 0: #only split rewards if the task is worth doing, otherwise get the full handover penalty
                rewards.append(beta_hat[i, chosen_task] / num_times_tasks_completed[chosen_task])
            else:
                rewards.append(beta_hat[i, chosen_task])
            
        #Update the prev assignment:
        self.curr_assignment = np.zeros((self.n, self.m))
        for i in range(self.n):
            self.curr_assignment[i, assignments[i]] = 1

        self.k += 1        

        self._obs = [self.curr_assignment[i,:] for i in range(self.n)]
        for l in range(self.L):
            if self.k + l < self.T:
                self._obs = [np.concatenate([self._obs[i], self.sat_prox_mat[i,:,self.k+l]]) for i in range(self.n)]
            else: #if the episode has ended within your lookahead of L, just pad with zeros
                self._obs = [np.concatenate([self._obs[i], np.zeros(self.m)]) for i in range(self.n)]

        done = self.k >= self.T

        if not done:
            self.beta = self.sat_prox_mat[:,:,self.k]
        else:
            self.beta = np.zeros((self.n, self.m))
        self.prev_assigns = assignments

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

    def beta_hat(self, beta, prev_assigns):
        """
        Given the info contained in the state, 
        returns a matrix of benefits for each agent-task pair.

        If initially has a time dimension, outputs a time_dim x n x m x L matrix.
        """
        #if beta or prev_assigns is a tensor, convert to numpy
        if isinstance(beta, th.Tensor): beta = beta.numpy()
        if isinstance(prev_assigns, th.Tensor): prev_assigns = prev_assigns.numpy()
        #add two dimensions (batch and time) to the beta and prev_assigns if they are not already there
        beta_init_dim = beta.ndim
        if beta_init_dim == 2:
            # beta = np.expand_dims(beta, axis=0)
            beta = np.expand_dims(beta, axis=0)
        prev_assigns_init_dim = prev_assigns.ndim
        if prev_assigns_init_dim == 1: 
            # prev_assigns = np.expand_dims(prev_assigns, axis=0)
            prev_assigns = np.expand_dims(prev_assigns, axis=0)
        # batch_dim = beta.shape[0]
        time_dim = beta.shape[0]

        prev_assign_mat = np.zeros((time_dim, self.n, self.m))
        # for batch in range(batch_dim):
        for time in range(time_dim):
            for i in range(self.n):
                prev_assign_mat[time, i, prev_assigns[time, i]] = 1

        #~~~~~~~~~~~~~~~~~~~~~~~~ CALCULATE HANDOVER PENALTY ~~~~~~~~~~~~~~~~~~~~~~~~
        #Calculate which transitions are not penalized because of the type of task they are
        #(i.e. some tasks are dummy tasks, or correspond to the same task at a different priority, etc.)
        #this info is held in T_trans.
        pen_locs_based_on_T_trans = prev_assign_mat @ self.T_trans

        #Calculate which transitions are not penalized because they are not meaningful tasks (i.e. have zero benefit)
        pen_locs_based_on_task_prios = beta > 1e-12

        #Penalty is only applied when both conditions are met, so multiply them elementwise
        penalty_locations = pen_locs_based_on_T_trans * pen_locs_based_on_task_prios

        #~~~~~~~~~~~~~~~~~~~~~~~~~~ APPLY HANDOVER PENALTY ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        beta_hat = beta.copy()
        beta_hat = beta_hat-self.lambda_*penalty_locations

        if beta_init_dim == 2: beta_hat = np.squeeze(beta_hat, axis=0)
        if prev_assigns_init_dim == 1: prev_assigns = np.squeeze(prev_assigns, axis=0)
        return beta_hat

def generate_benefits_over_time(n, m, T, width_min, width_max, scale_min=0.25, scale_max=2):
    """
    lightweight way of generating "constellation-like" benefit matrices.
    """
    benefits = np.zeros((n,m,T))
    for j in range(m):
        benefit_scale = np.random.choice([1, 1, 1, 10])
        for i in range(n):
            #Determine if task is active for this sat ever
            task_active = 1 if np.random.rand() > 0.75 else 0

            if task_active:
                #where is the benefit curve maximized
                time_center = np.random.uniform(0, T)

                #how wide is the benefit curve
                time_spread = np.random.uniform(width_min, width_max)
                sigma_2 = np.sqrt(time_spread**2/-8/np.log(0.05))
                  
                #iterate from time zero to t_final with T steps in between
                for t in range(T):
                    #calculate the benefit at time t
                    benefits[i,j,t] = benefit_scale*np.exp(-(t-time_center)**2/sigma_2/2)
    return benefits