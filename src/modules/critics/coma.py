import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.m = args.m
        self.n = args.n

        self.scheme = scheme
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.m)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state assumes (bs x max_time x state_dim, transformed to batch x time x n x state_dim)
        state = []
        for key in self.scheme.keys():
            if self.scheme[key].get("part_of_state", False):
                state.append(batch[key].view(bs, max_t, 1, -1).repeat(1, 1, self.n, 1))
        state = th.cat(state, dim=-1)
        inputs.append(state)

        # observation
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        print("actions", batch["actions_onehot"][0, 0,:,:])
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n, 1)
        agent_mask = (1 - th.eye(self.n, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.m).view(self.n, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        print("inputs", (actions * agent_mask.unsqueeze(0).unsqueeze(0))[0,0,:,:])

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

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = 0
        # get flattened size of state
        for key in scheme.keys():
            if scheme[key].get("part_of_state", False):
                shape_contribution = 1
                for dim in scheme[key]["vshape"]:
                    shape_contribution *= dim
                input_shape += shape_contribution
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n
        # last action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n
        return input_shape