import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
import numpy as np

class FilteredQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        if args.use_mps:
            device = "mps"
        elif args.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
            
        self.n = args.n
        self.m = args.m

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(self.n,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["rewards"][:, :-1].float()
        actions = batch["actions"][:, :-1].to(th.int64)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"].to(th.int64)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        total_beta = batch["beta"].float().sum(axis=-1)
        top_agent_tasks = th.topk(total_beta, k=self.args.env_args['M'], dim=-1).indices

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        #Pop the last column off of agent outputs for the no-op action, others are the Q-Values for top M actions
        baseline_action_benefit = mac_out[:, :, :, -1]
        top_M_q_values = mac_out[:, :, :, :-1]

        #Create a matrix where the default value is the baseline action benefit
        q_values_out = baseline_action_benefit.unsqueeze(-1).expand(-1, -1, -1, self.args.m).clone()

        #find M max indices in total_agent_benefits_by_task
        indices = th.tensor(np.indices(top_agent_tasks.shape))
        q_values_out[indices[0], indices[1], indices[2], top_agent_tasks] = top_M_q_values

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(q_values_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timestep's Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        #Pop the last column off of agent outputs for the no-op action, others are the Q-Values for top M actions
        target_baseline_action_benefit = target_mac_out[:, :, :, -1]
        target_top_M_q_values = target_mac_out[:, :, :, :-1]

        #Create a matrix where the default value is the baseline action benefit
        target_q_values_out = target_baseline_action_benefit.unsqueeze(-1).expand(-1, -1, -1, self.args.m).clone()

        #find M max indices in total_agent_benefits_by_task
        indices = th.tensor(np.indices(top_agent_tasks[:, 1:].shape))
        target_q_values_out[indices[0], indices[1], indices[2], top_agent_tasks[:, 1:]] = target_top_M_q_values

        # Mask out unavailable actions
        target_q_values_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            q_values_out_detach = q_values_out.clone().detach()
            q_values_out_detach[avail_actions == 0] = -9999999

            cur_max_actions = q_values_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_q_values_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_q_values_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if not self.args.use_mps_action_selection:
            self.mac.update_action_selector_agent()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n), t_env)
            self.log_stats_t = t_env

            avg_num_conflicts = self.calc_conflicting_actions(actions)
            self.logger.log_stat("avg_num_conflicts", avg_num_conflicts, t_env)

            avg_beta = self.calc_raw_benefits(batch["beta"], actions)
            self.logger.log_stat("avg_beta", avg_beta, t_env)

    def calc_conflicting_actions(self, actions):
        batches = actions.shape[0]
        timesteps = actions.shape[1]

        num_duplicates = 0
        for b in range(batches):
            for k in range(timesteps):
                num_times_done = np.zeros(self.m)
                for i in range(self.n):
                    chosen_action = actions[b, k, i, 0].item()
                    num_times_done[chosen_action] += 1
                
                num_duplicates += np.where(num_times_done > 0, num_times_done - 1, 0).sum()

        return num_duplicates/batches/timesteps

    def calc_raw_benefits(self, beta, actions):
        if beta.ndim == 4:
            pass
        elif beta.ndim == 5:
            beta = beta[:, :, :, :, 0]
        else:
            raise ValueError("beta has unexpected shape.", beta.shape)
        
        batches = actions.shape[0]
        timesteps = actions.shape[1]

        total_benefit = 0
        for b in range(batches):
            for k in range(timesteps):
                for i in range(self.n):
                    chosen_action = actions[b, k, i, 0].item()
                    total_benefit += beta[b, k, i, chosen_action]
                
        return total_benefit/batches/timesteps/self.n

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
    
    def mps(self):
        self.mac.agent.to("mps")
        self.target_mac.agent.to("mps")
        if self.mixer is not None:
            self.mixer.to("mps")
            self.target_mixer.to("mps")

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        print("LOADING FROM: ", path)
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
