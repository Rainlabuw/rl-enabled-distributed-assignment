import torch as th
from torch.distributions import Categorical
from components.epsilon_schedules import DecayThenFlatSchedule
import scipy.optimize
import numpy as np

class EpsilonGreedySAPTestActionSelector():
    """
    Epsilon greedy for training, but constrained to assignment problems for testing.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, beta=None):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        num_batches = agent_inputs.shape[0]
        n = agent_inputs.shape[1]
        m = agent_inputs.shape[2]

        if test_mode:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
            for batch in range(num_batches):
                # Solve the assignment problem for each batch, converting to numpy first
                benefit_matrix_from_q_values = agent_inputs[batch, :, :].detach().cpu()

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values, maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)
            return picked_actions
        else:
            if np.random.rand() < self.epsilon:
                return th.randperm(n) #TODO: Fix longterm, this does not select actions > m
            else:
                # mask actions that are excluded from selection
                masked_q_values = agent_inputs.clone()
                masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

                random_numbers = th.rand_like(agent_inputs[:, :, 0])
                pick_random = (random_numbers < self.epsilon).long()
                random_actions = Categorical(avail_actions.float()).sample().long()

                picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
                return picked_actions



class SequentialAssignmentProblemSelector():
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
    
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, beta=None):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        num_batches = agent_inputs.shape[0]
        n = agent_inputs.shape[1]
        m = agent_inputs.shape[2]

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n, device="cpu")
        for batch in range(num_batches):
            # if np.random.rand() < self.epsilon:
            #     # Solve the assignment problem for each batch, moving to cpu first
            #     picked_actions[batch, :] = th.randperm(n)

            # else:
            # Solve the assignment problem for each batch, moving to cpu first
            benefit_matrix_from_q_values = agent_inputs[batch, :, :].detach().cpu()

            #Add random noise
            avg_q_val = th.mean(th.abs(benefit_matrix_from_q_values))
            stds = th.ones_like(benefit_matrix_from_q_values)*avg_q_val*self.epsilon*2
            benefit_matrix_from_q_values += th.normal(mean=th.zeros_like(benefit_matrix_from_q_values), std=stds)

            _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values, maximize=True)
            picked_actions[batch, :] = th.tensor(col_ind)

            # print("benefit_matrix_from_q_values, eps", self.epsilon)
            # print(benefit_matrix_from_q_values)
            # # print("noise")
            # # print(th.normal(mean=th.zeros_like(benefit_matrix_from_q_values), std=stds))

        return picked_actions