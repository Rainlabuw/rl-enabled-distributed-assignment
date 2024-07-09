from functools import partial
from envs.constellation_env import ConstellationEnv
from envs.dictator_env import DictatorEnv
from envs.multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["constellation_env"] = partial(env_fn, env=ConstellationEnv)
REGISTRY["dictator_env"] = partial(env_fn, env=DictatorEnv)