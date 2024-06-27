from modules.agents import REGISTRY as agent_REGISTRY
from action_selectors import REGISTRY as action_REGISTRY
from action_selectors.non_rl_selectors import REGISTRY as non_rl_action_REGISTRY
from components.epsilon_schedules import DecayThenFlatSchedule
import torch as th
from scipy.special import softmax
import numpy as np

# This multi-agent controller shares parameters between agents
class JumpstartMAC:
    def __init__(self, scheme, groups, args):
        self.n = args.n
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.jumpstart_action_selector = non_rl_action_REGISTRY[args.jumpstart_action_selector](args)

        self.jumpstart_eps_schedule = DecayThenFlatSchedule(args.jumpstart_epsilon_start, args.jumpstart_epsilon_finish, args.jumpstart_epsilon_anneal_time,
                                              decay="linear")
        self.jumpstart_epsilon = self.jumpstart_eps_schedule.eval(0)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        self.jumpstart_epsilon = self.jumpstart_eps_schedule.eval(t_env)

        if test_mode:
            self.jumpstart_epsilon = self.args.jumpstart_evaluation_epsilon

        if np.random.rand() < self.jumpstart_epsilon:
            return self.jumpstart_action_selector.select_action(ep_batch[bs, t_ep])
        # Only select actions for the selected batch elements in bs
        else:
            avail_actions = ep_batch["avail_actions"][:, t_ep]
            # print("state: ", np.argmax(ep_batch["obs"][:, t_ep][0]))
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, action_selection_mode=True)
            chosem = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode, beta=ep_batch["beta"][bs, t_ep])
            return chosem

    def forward(self, ep_batch, t, test_mode=False, action_selection_mode=False):
        """
        If in action selector mode, rather than in training mode, use the selector agent.
        This is located on the CPU so computations are faster.
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if action_selection_mode and not self.args.use_mps_action_selection:
            agent_outs, self.hidden_states = self.selector_agent(agent_inputs, self.hidden_states)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states) #note that hidden state is unused if RNN not used

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            
            # if self.args.use_mps_action_selection:
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            # else: #otherwise, take softmax such that it remains on cpu
            #     agent_outs = softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.update_action_selector_agent() #ensure selector agent has same weights

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args) #Agent for training, potentially on GPU
        if not self.args.use_mps_action_selection:
            self.selector_agent = agent_REGISTRY[self.args.agent](input_shape, self.args) #Agent for selecting actions, always on CPU

    def update_action_selector_agent(self):
        self.selector_agent.load_state_dict(self.agent.state_dict())

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t].float())  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n

        return input_shape
