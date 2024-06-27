from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time
from collections import defaultdict

class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1, "EpisodeRunner only supports batch size 1"

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.T = self.env.T
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.T + 1,
                                preprocess=preprocess, device="cpu")
        self.mac = mac

        #Add environment to action selector (add in a list to be consistent w parallel)
        self.mac.action_selector.envs = [self.env]
        if self.args.mac == "jumpstart_mac":
            self.mac.jumpstart_action_selector.envs = [self.env]

    def get_env_info(self):
        return self.env.get_env_info()
    
    def get_env(self):
        return self.env

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset() #empty batch, reset environment, t=0
        terminated = False
        episode_return = 0
        #Initialize RNN hidden state for each agent
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = self.env.get_pretransition_data()

            #Add info to replay buffer that is available BEFORE the transition
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            rewards, terminated, env_info = self.env.step(actions[0])
            if test_mode and self.args.render:
                self.env.render()
            
            episode_return += sum(rewards)

            post_transition_data = {
                "actions": actions,
                "rewards": [(rewards,)],
                "terminated": [(terminated != env_info.get("T", False),)],
            }

            #Add info to replay buffer that is available AFTER the transition
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = self.env.get_pretransition_data()
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state (commenting out: why do we need this?)
        # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
            self.logger.log_stat("steps", self.t_env, self.t_env)

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()