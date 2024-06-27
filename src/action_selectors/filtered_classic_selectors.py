import torch as th
from torch.distributions import Categorical
from components.epsilon_schedules import DecayThenFlatSchedule
import numpy as np

class FilteredEpsilonGreedyActionSelector():
    """
    Epsilon greedy for training, but filtered to only top tasks.
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
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n, device="cpu")

        #General strategy: build a matrix where the default value is the baseline action benefit, then replace the M best actions with the M best Q-values.
        #Then select greedy actions from this matrix, unless the random number is under epsilon.
        top_M_benefits_from_q_values = agent_inputs[:, :, :].detach()

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
    
class FilteredSoftPoliciesSelector():
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, beta=None):
        assert beta is not None, "Need beta to figure out which are the top M tasks for each agent."
        if self.args.use_mps_action_selection:
            action_select_device = self.args.device
        else:
            action_select_device = "cpu"

        #Pick the indices of the top actions you're picking
        action_dist = Categorical(agent_inputs)
        picked_top_action_indices = action_dist.sample().long()

        total_beta = beta.sum(axis=-1)

        num_batches = beta.shape[0]
        n = beta.shape[1]
        m = beta.shape[2]

        #Translate top actions to real selections
        #find M max indices in total_agent_benefits_by_task, and random tasks for the baseline (axis -1 index M).
        top_agent_tasks = th.topk(total_beta, k=self.args.env_args['M'], dim=-1).indices

        #get a random matrix, make sure top M tasks are lower than everything else, and then take the argmax to get random tasks not in the top M
        random_choices = th.rand((num_batches, n, m), device=action_select_device)
        top_agent_tasks_mask = th.zeros((num_batches, n, m), device=action_select_device, dtype=th.bool)
        top_agent_tasks_mask.scatter_(-1, top_agent_tasks, True)
        masked_random_choices = random_choices.masked_fill(top_agent_tasks_mask, -1)
        random_baseline_tasks = masked_random_choices.argmax(dim=-1, keepdim=True)

        top_agent_tasks = th.cat((top_agent_tasks, random_baseline_tasks), dim=2)

        indices = th.tensor(np.indices(picked_top_action_indices.shape))
        picked_action_indices = top_agent_tasks[indices[0], indices[1], picked_top_action_indices]

        return picked_action_indices
    
# class FilteredMultinomialActionSelector():
#     #NOT IMPLEMENTED YET
#     def __init__(self, args):
#         self.args = args

#         self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
#                                               decay="linear")
#         self.epsilon = self.schedule.eval(0)
#         self.test_greedy = getattr(args, "test_greedy", True)

#     def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, beta=None):
#         masked_policies = agent_inputs.clone()
#         masked_policies[avail_actions == 0.0] = 0.0

#         self.epsilon = self.schedule.eval(t_env)

#         if test_mode and self.test_greedy:
#             picked_actions = masked_policies.max(dim=2)[1]
#         else:
#             picked_actions = Categorical(masked_policies).sample().long()

#         return picked_actions