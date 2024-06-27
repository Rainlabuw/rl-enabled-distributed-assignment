import torch as th
from components.epsilon_schedules import DecayThenFlatSchedule

class ContinuousActionSelector():
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.variance = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
        if getattr(self.args, "softmax_agent_inputs", False): agent_inputs = th.softmax(agent_inputs, dim=1)

        if test_mode:
            self.variance = self.args.evaluation_epsilon
            picked_actions = th.normal(agent_inputs, self.variance, device=self.args.device)
        else:
            self.variance = self.schedule.eval(t_env)
            picked_actions = th.normal(agent_inputs, self.variance, device=self.args.device)
        return picked_actions.detach()
    
    def action_log_prob(self, actions, old_agent_inputs):
        return th.distributions.Normal(old_agent_inputs, self.variance).log_prob(actions)