import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(ACCritic, self).__init__()

        self.args = args
        self.m = args.m
        self.n = args.n

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if not self.args.env_args.get("bids_as_actions", False):
            self.fc3 = nn.Linear(args.hidden_dim, 1)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.m)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(self.n, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # agent id
        input_shape += self.n
        return input_shape
