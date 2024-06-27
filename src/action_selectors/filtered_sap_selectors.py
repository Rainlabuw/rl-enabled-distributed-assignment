import torch as th
from components.epsilon_schedules import DecayThenFlatSchedule
from torch.distributions import Categorical
import scipy.optimize
import numpy as np

class FilteredSAPActionSelector():
    """
    SAP action selector, but with an exploration strategy that
    adds noise to the Q-values according to a variance given by epsilon.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
    
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, beta=None):
        assert beta is not None, "Need beta to figure out which are the top M tasks for each agent."
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        total_beta = beta.sum(axis=-1)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        num_batches = beta.shape[0]
        n = beta.shape[1]
        m = beta.shape[2]

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n, device="cpu")
        
        for batch in range(num_batches):
            # Solve the assignment problem for each batch, converting to numpy first
            top_M_benefits_from_q_values = agent_inputs[batch, :, :].detach().cpu()

            #Pop the last column off of top_M_benefits_from_q_values
            baseline_action_benefit = top_M_benefits_from_q_values[:, -1]
            top_M_benefits_from_q_values = top_M_benefits_from_q_values[:, :-1]

            #Create a matrix where the default value is the baseline action benefit
            benefit_matrix_from_q_values = baseline_action_benefit.unsqueeze(1).expand(n, m).clone()
            benefit_matrix_from_q_values += th.rand_like(benefit_matrix_from_q_values)*1e-8 #add noise to break ties between "do-nothing" actions randomly

            #find M max indices in total_agent_benefits_by_task
            top_agent_tasks = th.topk(total_beta[batch, :, :], k=self.args.env_args['M'], dim=-1).indices

            #find M max indices in total_agent_benefits_by_task
            indices = th.tensor(np.indices(top_agent_tasks.shape))
            benefit_matrix_from_q_values[indices[0], top_agent_tasks] = top_M_benefits_from_q_values

            #Add zero-mean gaussian noise with variance epsilon to the Q-values
            avg_q_val = th.mean(th.abs(benefit_matrix_from_q_values))
            stds = th.ones_like(benefit_matrix_from_q_values)*avg_q_val*self.epsilon*2
            benefit_matrix_from_q_values += th.normal(mean=th.zeros_like(benefit_matrix_from_q_values), std=stds)

            _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values.cpu(), maximize=True)
            picked_actions[batch, :] = th.tensor(col_ind)

        return picked_actions

class FilteredEpsGrSAPTestActionSelector():
    """
    Epsilon greedy for training, but constrained to assignment problems for testing.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, beta=None):
        assert beta is not None, "Need beta to figure out which are the top M tasks for each agent."
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        total_beta = beta.sum(axis=-1)

        num_batches = beta.shape[0]
        n = beta.shape[1]
        m = beta.shape[2]

        if test_mode:
            if self.args.use_mps_action_selection:
                picked_actions = th.zeros(num_batches, n, device=self.args.device)
            else:
                picked_actions = th.zeros(num_batches, n, device="cpu")
            for batch in range(num_batches): #need to split up by batch bc of linear_sum_assignment
                # Solve the assignment problem for each batch, converting to numpy first
                top_M_benefits_from_q_values = agent_inputs[batch, :, :].detach().cpu()

                #Pop the last column off of top_M_benefits_from_q_values
                baseline_action_benefit = top_M_benefits_from_q_values[:, -1]
                top_M_benefits_from_q_values = top_M_benefits_from_q_values[:, :-1]

                #Create a matrix where the default value is the baseline action benefit
                benefit_matrix_from_q_values = baseline_action_benefit.unsqueeze(1).expand(n, m).clone()
                benefit_matrix_from_q_values += th.rand_like(benefit_matrix_from_q_values)*1e-8 #add noise to break ties between "do-nothing" actions randomly

                #find M max indices in total_agent_benefits_by_task
                top_agent_tasks = th.topk(total_beta[batch, :, :], k=self.args.env_args['M'], dim=-1).indices

                #find M max indices in total_agent_benefits_by_task
                indices = th.tensor(np.indices(top_agent_tasks.shape))
                benefit_matrix_from_q_values[indices[0], top_agent_tasks] = top_M_benefits_from_q_values

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values.cpu(), maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)
            return picked_actions
        else:
            if self.args.use_mps_action_selection:
                picked_actions = th.zeros(num_batches, n, device=self.args.device)
            else:
                picked_actions = th.zeros(num_batches, n, device="cpu")

            #General strategy: build a matrix where the default value is the baseline action benefit, then replace the M best actions with the M best Q-values.
            #Then select greedy actions from this matrix, unless the random number is under epsilon.
            top_M_benefits_from_q_values = agent_inputs[:, :, :].detach().cpu()

            #Pop the last column off of top_M_benefits_from_q_values
            baseline_action_benefit = top_M_benefits_from_q_values[:, :, -1]
            top_M_benefits_from_q_values = top_M_benefits_from_q_values[:, :, :-1]

            #Create a matrix where the default value is the baseline action benefit
            benefit_matrix_from_q_values = baseline_action_benefit.unsqueeze(2).expand(num_batches, n, m).clone()
            benefit_matrix_from_q_values += th.rand_like(benefit_matrix_from_q_values)*1e-8 #add noise to break ties between "do-nothing" actions randomly

            #find M max indices in total_agent_benefits_by_task
            top_agent_tasks = th.topk(total_beta, k=self.args.env_args['M'], dim=-1).indices

            #find M max indices in total_agent_benefits_by_task
            indices = th.tensor(np.indices(top_agent_tasks.shape))
            benefit_matrix_from_q_values[indices[0], indices[1], top_agent_tasks] = top_M_benefits_from_q_values

            #Pick random actions with probability epsilon
            random_numbers = th.rand_like(agent_inputs[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()

            picked_actions = pick_random * random_actions + (1 - pick_random) * benefit_matrix_from_q_values.max(dim=2)[1]

            return picked_actions