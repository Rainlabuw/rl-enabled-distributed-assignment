import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd
import numpy as np

class FilteredCOMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n = args.n
        self.m = args.m
        self.M = args.env_args['M']
        self.N = args.env_args['N']
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        if args.use_mps:
            device = "mps"
        elif args.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
            
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n,), device=device)

        # in filtered COMA, each agent does have their own specific rewards because they get rewarded based on their neighborhood
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(self.n,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["rewards"][:, :-1].float()
        actions = batch["actions"][:, :].to(th.int64)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = th.ones((bs, max_t-1, self.n, self.M+1)) #hardcode all M+1 actions to be available

        total_beta = batch["beta"].float().sum(axis=-1)
        top_agent_tasks = th.topk(total_beta, k=self.M, dim=-1).indices

        neighboring_rewards = th.zeros((bs, max_t-1, self.n), device=self.args.device)
        neighbors = th.zeros((bs, max_t, self.n, self.N), device=self.args.device, dtype=th.long) #neighbors for each agent at each timestep
        #Each agent gets the rewards from the sum of the neighboring N agents
        for i in range(self.n):
            #find M max indices in total_agent_benefits
            top_agenti_tasks = top_agent_tasks[:,:,i,:] # b x t x M
            top_agenti_tasks_expanded = top_agenti_tasks.unsqueeze(2).repeat(1,1,self.n,1) # b x t x n x M

            #Determine the N agents who most directly compete with agent i
            # (i.e. the N agents with the highest total benefit for the top M tasks)
            top_M_indices = np.indices(top_agenti_tasks_expanded.shape)
            total_benefits_for_top_M_tasks = total_beta[top_M_indices[0], top_M_indices[1], top_M_indices[2], top_agenti_tasks_expanded] # b x t x n x M
            best_task_benefit_by_agent, _ = th.max(total_benefits_for_top_M_tasks, dim=-1) # b x t x n
            best_task_benefit_by_agent[:, :, i] = -th.inf #set agent i to a really low value so it doesn't show up in the sort
            top_N = th.topk(best_task_benefit_by_agent, k=self.N, dim=-1).indices # b x t x N (N agents which have the highest value for a task in the top M for agent i)

            top_N_indices = np.indices(top_N[:,:-1].shape)
            neighboring_rewards[:, :, i] = rewards[top_N_indices[0], top_N_indices[1], top_N[:,:-1]].sum(axis=-1)
            neighbors[:,:,i,:] = top_N

            # if i == 0:
            #     for n in range(self.N):
            #         nn = top_N[0,0,n].item()
            #         print("agent ", nn, best_task_benefit_by_agent[0,0,nn])
            #     print("top agent 0 tasks", top_agent_tasks[0,0,0,:])
            #     print("agent 0 neighbors", top_N[0,0,:])
            #     print("best_task_benefits", best_task_benefit_by_agent[0,0,top_N[0,0,:]])
            #     print("best_real_task_ben", best_task_benefit_by_agent[0,0,top_N[0,0,:]][0])

        # map actions to top M actions
        actions_one_hot = F.one_hot(actions.squeeze(-1), num_classes=self.m)

        top_M_indices = np.indices(top_agent_tasks.shape)
        top_M_actions_onehot = actions_one_hot[top_M_indices[0], top_M_indices[1], top_M_indices[2], top_agent_tasks]
        did_agent_not_do_a_top_M_task = (top_M_actions_onehot.sum(axis=-1) == 0).unsqueeze(-1)
        top_Mp1_actions_onehot = th.cat([top_M_actions_onehot, did_agent_not_do_a_top_M_task], dim=-1)
        top_Mp1_actions = th.argmax(top_Mp1_actions_onehot, dim=-1).unsqueeze(-1) #to match the shape of original actions


        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, top_Mp1_actions, avail_actions,
                                                        critic_mask, bs, max_t, neighbors, top_agent_tasks)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.M+1)
        pi = mac_out.view(-1, self.M+1)
        baseline = (pi * q_vals).sum(-1).detach() #expected value of all Q-values

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=top_Mp1_actions[:,:-1].reshape(-1, 1)).squeeze(1)
        pi_taken = th.gather(pi, dim=1, index=top_Mp1_actions[:,:-1].reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        
        coma_loss = - ((advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_training_steps += 1

        if self.args.target_update_interval_or_tau > 1 and (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if not self.args.use_mps_action_selection:
            self.mac.update_action_selector_agent()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

            avg_num_conflicts = self.calc_conflicting_actions(actions[:,:-1])
            self.logger.log_stat("avg_num_conflicts", avg_num_conflicts, t_env)

            avg_beta = self.calc_raw_benefits(batch["beta"], actions[:,:-1])
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
            raise ValueError(f"beta has unexpected shape, {beta.shape}")
        
        batches = actions.shape[0]
        timesteps = actions.shape[1]

        total_benefit = 0
        for b in range(batches):
            for k in range(timesteps):
                for i in range(self.n):
                    chosen_action = actions[b, k, i, 0].item()
                    total_benefit += beta[b, k, i, chosen_action]
                
        return total_benefit/batches/timesteps/self.n

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t, 
                      neighbors, top_agent_tasks):
        # Optimise critic
        with th.no_grad():
            target_q_vals = self.target_critic(batch, neighbors, top_agent_tasks)

        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        if self.args.standardise_returns:
            targets_taken = targets_taken * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = self.nstep_returns(rewards, mask, targets_taken, self.args.q_nstep)

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        actions = actions[:, :-1]
        q_vals = self.critic(batch, neighbors, top_agent_tasks)[:, :-1]
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)

        td_error = (q_taken - targets.detach())
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((q_taken * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((targets * mask).sum().item() / mask_elems)

        return q_vals, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def mps(self):
        self.mac.agent.to("mps")
        self.critic.to("mps")
        self.target_critic.to("mps")

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
