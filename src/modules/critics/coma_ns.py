import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP


class COMACriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(COMACriticNS, self).__init__()

        self.args = args
        self.m = args.m
        self.n = args.n

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.critics = [MLP(input_shape, args.hidden_dim, self.m) for _ in range(self.n)]

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n):
            q = self.critics[i](inputs[:, :, i]).unsqueeze(2)
            qs.append(q)
        return th.cat(qs, dim=2)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n, 1))

        # observation
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n, 1)
        agent_mask = (1 - th.eye(self.n, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.m).view(self.n, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n, 1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n, 1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n, 1)
                inputs.append(last_actions)

        inputs = th.cat([x.reshape(bs, max_t, self.n, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]

        # actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n

        # last action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n
        # agent id
        # input_shape += self.n
        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()