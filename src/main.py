import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import logging

from gym.envs.registration import register
import wandb

from run import run

import cProfile

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    return run(_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def experiment_run(params, explicit_dict_items=None, verbose=True):
    if not verbose: 
        ex.logger.setLevel(logging.ERROR)
        sys.stdout = open(os.devnull, 'w')

    #Determine if we want to repeat the experiment
    num_repeats = 1
    param_copy = deepcopy(params)
    for p in param_copy:
        if p.startswith("--repeat"):
            num_repeats = int(p.split("=")[1])
            params.remove(p)

    th.set_num_threads(1)

    # Get the default configs from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    #Add the env and algorithm config to the default configs
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    #Add items from explicit_dict_items (hardcoded for 2 levels of dicts)
    if explicit_dict_items is not None:
        for k, v in explicit_dict_items.items():
            if type(v) == dict:
                for k2, v2 in v.items():
                    config_dict[k][k2] = v2
            else:
                config_dict[k] = v

    ex.add_config(config_dict)

    for exp_num in range(num_repeats):
        print(f"Starting experiment {exp_num+1}/{num_repeats}")
        experiment_object = ex.run_commandline(params)

    #set stdout back to normal
    if not verbose: sys.stdout = sys.__stdout__

    return experiment_object

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    experiment_run(params)