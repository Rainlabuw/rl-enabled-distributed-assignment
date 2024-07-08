from functools import partial
from envs.real_power_constellation_env import RealPowerConstellationEnv
from envs.dictator_env import DictatorEnv
from envs.multiagentenv import MultiAgentEnv
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["real_power_constellation_env"] = partial(env_fn, env=RealPowerConstellationEnv)
REGISTRY["dictator_env"] = partial(env_fn, env=DictatorEnv)