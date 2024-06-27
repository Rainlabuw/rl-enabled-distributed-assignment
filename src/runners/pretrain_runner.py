from envs import REGISTRY as env_REGISTRY
from action_selectors.non_rl_selectors import REGISTRY as non_rl_selector_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import scipy.optimize
from collections import defaultdict
import time

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class PretrainRunner:
    def __init__(self, args, logger, buffer, scheme, groups):
        self.args = args
        self.logger = logger

        self.batch_size = self.args.pretrain_batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i
        self.sample_env = env_fn(**self.args.env_args)

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.T = self.args.env_args["T"]

        self.t = 0
        self.b = buffer.buffer_size

        self.buffer = buffer

        #Build pretraining function, and add envs to it
        self.pretrain_fn = non_rl_selector_REGISTRY[self.args.pretrain_fn](self.args)

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_env", None))
        envs = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.pretrain_fn.envs = envs

        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.T + 1,
                                preprocess=self.buffer.preprocess, device="cpu")

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

    def fill_buffer(self):
        """
        Fills buffer until full by running individual episodes with run().
        """
        while self.b < self.args.buffer_size:
            if (self.b % 100) == 0: self.logger.console_logger.info(f"Generating offline dataset: {self.b}/{self.args.buffer_size}")
            episode_batch = self.run()
            self.buffer.insert_episode_batch(episode_batch)

            self.b += self.batch_size

        return self.buffer

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # NOTE: will need to actually pass in env per batch if I use HAAL to generate databases.
            # NOTE: might break because of the timestep situation - betahat expects no timestep axis?
            actions = self.pretrain_fn.select_action(self.batch[:, self.t])

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
                        parent_conn.send(("step", actions[action_idx]))
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

                    env_terminated = False
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
        return self.batch

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