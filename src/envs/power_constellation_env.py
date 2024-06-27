import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import scipy.optimize

class PowerConstellationEnv(Env):
    """
    Model a constellation, where benefits are provided to you for task in several states
    """
    def __init__(self,
                 n, num_tasks, 
                 T, 
                 L, lambda_,
                 bids_as_actions=False,
                 seed=None,
                 sat_prox_mat=None,#if you want to train on only one specific benefit matrix
                 benefit_info=None):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        self.n = n
        self.m = num_tasks
        self.T = T
        print(f"Episode step limit: {self.T}")

        self.benefit_fn = meaningful_task_handover_pen_benefit_fn
        self.benefit_info = benefit_info

        if sat_prox_mat is None:
            self.constant_benefits = False
            self.sat_prox_mat = generate_benefits_over_time(n, num_tasks, T, 3, 6)
        else:
            self.constant_benefits = True
            self.sat_prox_mat = sat_prox_mat

        #Number of steps to look into the future
        self.L = L
        self.lambda_ = lambda_
        self.graphs = None #both hardcoded to None for now
        self.init_assignment = None

        #State is the previous assignment, plus the power state of the satellite
        self.curr_assignment = np.zeros((self.n, self.m))
        self.power_states = np.ones(self.n)

        self.k = 0

        self.bids_as_actions = bids_as_actions
        if self.bids_as_actions:
            #Action space is a bid for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=0, high=np.inf, shape=(self.m,))] * self.n))
        else:
            #Action space is one action for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.m)] * self.n))

        #Observation space is just the benefits per task for the next L time steps, plus the agent's previous assignment, plus the power state.
        self.obs_space_size = self.L * self.m + self.m + 1
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,))] * self.n))

    def step(self, actions):
        """ 
        Input is actions in Discrete form.
        Returns reward, terminated, info.
        """
        # print("power", self.power_states[0])

        if self.bids_as_actions:
            _, assignments = scipy.optimize.linear_sum_assignment(actions, maximize=True)
        else:
            assignments = [int(a) for a in actions]

        adj_benefits = self.benefit_fn(self.sat_prox_mat[:,:,self.k], self.curr_assignment, self.lambda_)

        num_times_tasks_completed = np.zeros(self.m)
        for i in range(self.n):
            if self.power_states[i] > 0:
                num_times_tasks_completed[assignments[i]] += 1

        rewards = []
        for i in range(self.n):
            if self.power_states[i] > 0:
                chosen_task = assignments[i]
                if adj_benefits[i, chosen_task] > 0: #only split rewards if the task is worth doing, otherwise get the full handover penalty
                    rewards.append(adj_benefits[i, chosen_task] / num_times_tasks_completed[chosen_task])
                else:
                    rewards.append(adj_benefits[i, chosen_task])
            else:
                rewards.append(0)

        #Update the prev assignment:
        self.curr_assignment = np.zeros((self.n, self.m))
        for i in range(self.n):
            self.curr_assignment[i, assignments[i]] = 1

        #Update agent power states
        for i in range(self.n):
            if self.power_states[i] > 0:
                chosen_task = assignments[i]
                if self.sat_prox_mat[i,chosen_task,self.k] > 1e-12:
                    self.power_states[i] -= 0.2
                else:
                    self.power_states[i] = min(self.power_states[i] + 0.1, 1)
            else:
                # print(f"Agent {i} is dead")
                pass #do nothing if the agent is already dead

        # print("assign",self.sat_prox_mat[0,assignments[0],self.k])

        self.k += 1        

        self._obs = [self.curr_assignment[i,:] for i in range(self.n)]
        for l in range(self.L):
            if self.k + l < self.T:
                self._obs = [np.concatenate([self._obs[i], self.sat_prox_mat[i,:,self.k+l]]) for i in range(self.n)]
            else: #if the episode has ended within your lookahead of L, just pad with zeros
                self._obs = [np.concatenate([self._obs[i], np.zeros(self.m)]) for i in range(self.n)]
        self._obs = [np.concatenate([self._obs[i], [self.power_states[i]]]) for i in range(self.n)] #add power state

        done = self.k >= self.T
        return rewards, done, {}

    def reset(self):
        """ Resets environment."""
        self.curr_assignment = np.zeros((self.n, self.m))
        self.k = 0
        self.power_states = np.ones(self.n)

        if not self.constant_benefits:
            self.sat_prox_mat = generate_benefits_over_time(self.n, self.m, self.T, 3, 6)
        else:
            pass #if benefits are constant, do nothing because you should use the same benefit matrix

        self._obs = [self.curr_assignment[i,:] for i in range(self.n)]
        for l in range(self.L):
            if self.k + l < self.T:
                self._obs = [np.concatenate([self._obs[i], self.sat_prox_mat[i,:,self.k+l]]) for i in range(self.n)]
            else: #if the episode has ended within your lookahead of L, just pad with zeros
                self._obs = [np.concatenate([self._obs[i], np.zeros(self.m)]) for i in range(self.n)]
        self._obs = [np.concatenate([self._obs[i], [self.power_states[i]]]) for i in range(self.n)] #reset all agents to full power

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

def meaningful_task_handover_pen_benefit_fn(sat_prox_mat, prev_assign, lambda_, benefit_info=None):
    """
    Adjusts a 3D benefit matrix to account for generic handover penalty (i.e. constant penalty for switching tasks).

    benefit_info is an object which can contain extra information about the benefits - it should store:
     - task_benefits is a m-length array of the baseline benefits associated with each task.
     - T_trans is a matrix which determines which transitions between TASKS are penalized.
        It is m x m, where entry ij is the state dependence multiplier that should be applied when switching from task i to task j.
        (If it is None, then all transitions between different tasks are scaled by 1.)
    Then, prev_assign @ T_trans is the matrix which entries of the benefit matrix should be adjusted.
    """
    if lambda_ is None: lambda_ = 0 #if lambda_ is not provided, then add no penalty so lambda_=0
    init_dim = sat_prox_mat.ndim
    if init_dim == 2: sat_prox_mat = np.expand_dims(sat_prox_mat, axis=2)
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    L = sat_prox_mat.shape[2]

    #Generate a matrix which determines the benefits of each task at each timestep.
    if benefit_info is not None and benefit_info.task_benefits is not None:
        task_benefits = benefit_info.task_benefits
    else:
        task_benefits = np.ones(m)
    task_benefits = np.tile(task_benefits, (n,1))
    task_benefits = np.repeat(task_benefits[:,:,np.newaxis], L, axis=2)

    benefits_hat = sat_prox_mat * task_benefits

    #Determine tasks for which to apply a handover penalty.
    if prev_assign is None:
        penalty_locations = np.zeros((n,m))
    else:
        #Calculate which transitions are not penalized because of the type of task they are
        #(i.e. some tasks are dummy tasks, or correspond to the same task at a different priority, etc.)
        try:
            T_trans = benefit_info.T_trans
        except AttributeError:
            T_trans = None
        if T_trans is None:
            T_trans = np.ones((m,m)) - np.eye(m) #default to all transitions (except nontransitions) being penalized
        pen_locs_based_on_T_trans = prev_assign @ T_trans

        #Calculate which transitions are not penalized because they are not meaningful tasks (i.e. have zero benefit)
        pen_locs_based_on_task_benefits = benefits_hat.sum(-1) > 1e-12

        #Penalty is only applied when both conditions are met, so multiply them elementwise
        penalty_locations = pen_locs_based_on_T_trans * pen_locs_based_on_task_benefits

    benefits_hat[:,:,0] = benefits_hat[:,:,0]-lambda_*penalty_locations

    if init_dim == 2: 
        benefits_hat = np.squeeze(benefits_hat, axis=2)
    return benefits_hat

def generate_benefits_over_time(n, m, T, width_min, width_max, scale_min=0.25, scale_max=2):
    """
    lightweight way of generating "constellation-like" benefit matrices.
    """
    benefits = np.zeros((n,m,T))
    for i in range(n):
        for j in range(m):
            #Determine if task is active for this sat ever
            task_active = 1 if np.random.rand() > 0.75 else 0

            if task_active:
                #where is the benefit curve maximized
                time_center = np.random.uniform(0, T)

                #how wide is the benefit curve
                time_spread = np.random.uniform(width_min, width_max)
                sigma_2 = np.sqrt(time_spread**2/-8/np.log(0.05))

                #how high is the benefit curve
                benefit_scale = np.random.uniform(scale_min, scale_max)
                  
                #iterate from time zero to t_final with T steps in between
                for t in range(T):
                    #calculate the benefit at time t
                    benefits[i,j,t] = benefit_scale*np.exp(-(t-time_center)**2/sigma_2/2)
    return benefits