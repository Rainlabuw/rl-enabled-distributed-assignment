from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
from collections import defaultdict

import time
# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.T = self.args.env_args["T"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        if not self.args.use_mps_action_selection:
            self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.T + 1,
                                    preprocess=preprocess, device="cpu")
        else:
            self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.T + 1,
                                    preprocess=preprocess, device=self.args.device)
        self.mac = mac
        
        # self.update_action_selector_envs()
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_env", None))
        envs = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.mac.action_selector.envs = envs
        if self.args.mac == "jumpstart_mac":
            self.mac.jumpstart_action_selector.envs = envs

        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
    
    def get_env(self):
        """
        Return the first environment in the batch, likely to be used as
        a reference for the action selector.
        """
        self.parent_conns[0].send(("get_env", None))
        return self.parent_conns[0].recv()
    
    def update_action_selector_envs(self):
        """
        Return all environments in the batch, and add them to the action selectors.

        TOO SLOW.
        """
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_env", None))
        envs = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.mac.action_selector.envs = envs
        self.mac.jumpstart_action_selector.envs = envs
    
    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = defaultdict(list)
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            for k, v in data.items():
                pre_transition_data[k].extend(v)

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        st = time.time()
        self.reset()
        print("Reset time: ", time.time() - st)

        st = time.time()
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # self.update_action_selector_envs() #TOO SLOW
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "rewards": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = defaultdict(list)

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    worker_post_transition_data = data[0]
                    worker_pre_transition_data = data[1]
                    # Remaining data for this current timestep
                    post_transition_data["rewards"].append((worker_post_transition_data["rewards"],))

                    if getattr(self.args, "cooperative_rewards", False):
                        episode_returns[idx] += worker_post_transition_data["rewards"]
                    else:
                        episode_returns[idx] += sum(worker_post_transition_data["rewards"])
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if post_transition_data["terminated"]:
                        final_env_infos.append(worker_post_transition_data["info"])
                    if post_transition_data["terminated"] and not worker_post_transition_data["info"].get("T", False):
                        env_terminated = True
                    terminated[idx] = worker_post_transition_data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    for k, v in worker_pre_transition_data.items():
                        pre_transition_data[k].extend(v)

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        if not test_mode:
            self.t_env += self.env_steps_this_run

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
            self.logger.log_stat("steps", self.t_env, self.t_env)
        print("Time to execute actions: ", time.time() - st)
        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            rewards, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            
            post_transition_data = {
                # Rest of the data for the current timestep
                "rewards": rewards,
                "terminated": terminated,
                "info": env_info
            }

            pretransition_data = env.get_pretransition_data()

            remote.send([post_transition_data, pretransition_data])
        elif cmd == "reset":
            env.reset()
            pretransition_data = env.get_pretransition_data()
            remote.send(pretransition_data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        elif cmd == "get_env":
            remote.send(env)
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

