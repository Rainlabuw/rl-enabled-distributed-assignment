import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np
import pickle

from runners.pretrain_runner import PretrainRunner

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, EpisodeBatch
from components.transforms import OneHot


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    if args.use_mps: args.device = "mps"
    elif args.use_cuda: args.device = "cuda"
    else: args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    #Don't print config if you're passing in benefits_over_time, because its too large printed
    if _config["env_args"].get("sat_prox_mat", None) is None:
        _log.info("Experiment Parameters:")
        experiment_params = pprint.pformat(_config, indent=4, width=1)
        _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    unique_token = f"{_config['name']}_seed{_config['seed']}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        if args.wandb_run_name == None: args.wandb_run_name = unique_token
        logger.setup_wandb(args)

    # sacred is off by default
    # logger.setup_sacred(_run)

    # Run and train
    if args.evaluate:
        if args.env == "real_power_constellation_env":
            actions, reward, ps = run_sequential(args=args, logger=logger)
        else:
            actions, reward = run_sequential(args=args, logger=logger)
    else:
        run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    if args.use_wandb:
        import wandb
        wandb.finish()

    # Making sure framework really exits
    # os._exit(os.EX_OK)
    if args.evaluate: 
        if args.env == "real_power_constellation_env":
            return actions, reward, ps
        else:
            return actions, reward

def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        batch = runner.run(test_mode=True)

    episode_actions = batch.data.transition_data['actions'][0,:,:,0]
    episode_reward = batch.data.transition_data['rewards'].sum()

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

    if args.env == "real_power_constellation_env":
        power_states = batch.data.transition_data['power_states'][0,-1,:]
        episode_power_states = batch.data.transition_data['power_states']
        #Return the actions taken by agents over the course of the episode
        return episode_actions, episode_reward, power_states
    else:
        return episode_actions, episode_reward


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    sample_env = runner.get_env()
    args.n = sample_env.n
    args.m = sample_env.m
    args.T = sample_env.T

    groups = {"agents": args.n}

    #~~~~~~~~SET UP BUFFER (either load existing buffer, or generate a new one)~~~~~~~~~
    use_bc = getattr(args, "use_bc", False)
    use_offline_rl = getattr(args, "use_offline_rl", False)
    if not use_bc and not use_offline_rl:
        logger.console_logger.info("No offline dataset desired - proceeding as normal.")
        buffer = ReplayBuffer(
            sample_env.scheme,
            groups,
            args.buffer_size,
            args.T + 1, #max_seq_length
            preprocess=sample_env.preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )
    else:
        if args.offline_dataset_path is None:
            logger.console_logger.info("No offline dataset found: generating one with {}".format(args.pretrain_fn))
            buffer = ReplayBuffer(
                sample_env.scheme,
                groups,
                args.buffer_size,
                args.T + 1, #max_seq_length
                preprocess=sample_env.preprocess,
                device="cpu", #always generate on the CPU
            )
            pretrain_runner = PretrainRunner(args, logger, buffer, sample_env.scheme, groups) #need to provide scheme and groups explicitly so that the scheme doesn't contain filled
            buffer = pretrain_runner.fill_buffer()
            with open(f"datasets/{args.unique_token}.pkl", 'wb') as f:
                pickle.dump(buffer, f)
            logger.console_logger.info("Done generating and saving offline dataset.")
        else:
            logger.console_logger.info("Offline dataset provided: loading from {}".format(args.offline_dataset_path))
            with open(f"datasets/{args.offline_dataset_path}.pkl", 'rb') as f:
                buffer = pickle.load(f)

            if buffer.buffer_size < args.buffer_size:
                logger.console_logger.info("Offline dataset is smaller than desired buffer size: generating more data with {}".format(args.pretrain_fn))
                pretrain_runner = PretrainRunner(args, logger, buffer, sample_env.scheme, groups)
                buffer = pretrain_runner.fill_buffer()
                with open(f"datasets/{args.unique_token}.pkl", 'wb') as f:
                    pickle.dump(buffer, f)

            logger.console_logger.info("Done loading offline dataset.")

    # ~~~~~~~~~~~~~~~~ SET UP MAC, LEARNER ~~~~~~~~~~~~~~~~
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme, create env and give it to the mac as well
    runner.setup(scheme=sample_env.scheme, groups=groups, preprocess=sample_env.preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if use_bc: bc_learner = le_REGISTRY["bc_learner"](mac, buffer.scheme, logger, args)

    if args.use_mps: 
        learner.mps()
        if use_bc: bc_learner.mps()
    elif args.use_cuda: 
        learner.cuda()
        if use_bc: bc_learner.cuda()


    # ~~~~~~~~~~~~~~~~ LOAD MODEL IF DESIRED ~~~~~~~~~~~~~~~~
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        skip_load = False
        checkpoint_path = os.path.join('/gscratch/socialrl/josh_holder/marl_sap/results/models', args.checkpoint_path)
        if not os.path.isdir(checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist, skipping checkpoint load".format(checkpoint_path)
            )
            skip_load = True

        if not skip_load:
            # Go through all files in args.checkpoint_path
            for name in os.listdir(checkpoint_path):
                full_name = os.path.join(checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

            model_path = os.path.join(checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path)
            if use_bc: bc_learner.load_models(model_path)
            runner.t_env = timestep_to_load

    # ~~~~~~~~~~~~~~~~ EVALUATE IF DESIRED (skip the rest of training if so) ~~~~~~~~~~~~~~~~
    if args.evaluate or args.save_replay:
        runner.log_train_stats_t = runner.t_env
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")

        if args.env == "real_power_constellation_env":
            actions, reward, ps = evaluate_sequential(args, runner)
            return actions, reward, ps
        else:
            actions, reward = evaluate_sequential(args, runner)
            return actions, reward

    # ~~~~~~~~~~~~~~~ COMPLETE PRETRAINING (BC or offline RL) IF DESIRED ~~~~~~~~~~~~~~~
    episode = 0
    last_test_T = -args.test_interval - 1
    # last_test_T = 0 #changing this for now so we get into training quicker
    last_log_T = 0
    model_save_time = 0

    if use_offline_rl:
        last_test_T, last_log_T, model_save_time, episode = run_offline_rl_pretraining(args, logger, runner, buffer, learner,
                                                                           last_test_T, last_log_T, model_save_time, episode)
        logger.console_logger.info("Done with offline RL pretraining after {} steps".format(runner.t_env))
    if use_bc:
        last_test_T, last_log_T, model_save_time, episode = run_behavior_cloning_pretraining(args, logger, runner, buffer, bc_learner,
                                                                            last_test_T, last_log_T, model_save_time, episode)
        #Reset the buffer to an empty buffer, with the size of the batch size.
        buffer = ReplayBuffer(
                sample_env.scheme,
                groups,
                args.batch_size, #policy gradient methods don't use a replay buffer, so just train on the most recent batch
                args.T + 1, #max_seq_length
                preprocess=sample_env.preprocess,
                device="cpu" if args.buffer_cpu_only else args.device,
            )
        logger.console_logger.info("Done with BC after {} steps".format(runner.t_env))

    # ~~~~~~~~~~~~~~~~ REAL TRAINING LOOP ~~~~~~~~~~~~~~~
    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} more timesteps".format(args.t_max - runner.t_env))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        st = time.time()
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            #If the data from the replay buffer is on CPU, move it to GPU
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            print("Training time: ", time.time() - st)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            # save_path = os.path.join(
            #     args.local_results_path, "models", args.unique_token, str(runner.t_env)
            # )
            # "results/models/{}".format(unique_token)
            save_path = os.path.join(
                args.local_results_path, "models", args.wandb_run_name, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def run_offline_rl_pretraining(args, logger, runner, buffer, learner,
                    last_test_T, last_log_T, model_save_time, episode):
    epochs = getattr(args, "epochs", 1) #if there are no epochs, usually there is one grad update for each batch

    pretrain_batches = runner.t_env * epochs // args.T
    while pretrain_batches < args.pretrain_batches:
        if (pretrain_batches % 50) == 0: logger.console_logger.info(f"Pretraining w offline RL, {pretrain_batches}/{args.pretrain_batches}")
        episode_sample = buffer.sample(args.batch_size)

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

        #If the data from the replay buffer is on CPU, move it to GPU
        if episode_sample.device != args.device:
            episode_sample.to(args.device)

        learner.train(episode_sample, 0, episode_num=0)
        pretrain_batches += args.batch_size_run // epochs

        #Increment runner t_env to simulate the number of environment steps
        #In normal training, training happens after each episode, so we need to increment t_env by the number of steps in an episode
        runner.t_env += args.batch_size_run * args.T // epochs

        # ~~~~~~~~~~~ LOGGING DURING OFFLINE RL ~~~~~~~~~~~
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            # save_path = os.path.join(
            #     args.local_results_path, "models", args.unique_token, str(runner.t_env)
            # )
            save_path = os.path.join(
                args.local_results_path, "models", args.wandb_run_name, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    return last_test_T, last_log_T, model_save_time, episode
    
def run_behavior_cloning_pretraining(args, logger, runner, buffer, learner,
                         last_test_T, last_log_T, model_save_time, episode):
    epochs = getattr(args, "epochs", 1) #if there are no epochs, usually there is one grad update for each batch

    pretrain_batches = runner.t_env * epochs // args.T
    while pretrain_batches < args.pretrain_batches:
        if (pretrain_batches % 50) == 0: logger.console_logger.info(f"Pretraining, {pretrain_batches}/{args.pretrain_batches}")
        episode_sample = buffer.sample(args.batch_size)

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

        #If the data from the replay buffer is on CPU, move it to GPU
        if episode_sample.device != args.device:
            episode_sample.to(args.device)

        st = time.time()
        learner.train(episode_sample, 0, episode_num=0)
        print(f"Pretrain training time: {time.time()-st}")
        pretrain_batches += args.batch_size_run // epochs

        #Increment runner t_env to simulate the number of environment steps
        #In normal training, training happens after each episode, so we need to increment t_env by the number of steps in an episode
        runner.t_env += args.batch_size_run * args.T // epochs

        # ~~~~~~~~~~~ LOGGING DURING BC ~~~~~~~~~~~
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            # save_path = os.path.join(
            #     args.local_results_path, "models", args.unique_token, str(runner.t_env)
            # )
            save_path = os.path.join(
                args.local_results_path, "models", args.wandb_run_name, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    return last_test_T, last_log_T, model_save_time, episode


def args_sanity_check(config, _log):
    # set MPS and CUDA flags
    if config["use_mps"] and not th.backends.mps.is_available():
        config["use_mps"] = False
        _log.warning(
            "MPS flag use_mps was switched OFF automatically because no MPS devices are available!"
        )
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
