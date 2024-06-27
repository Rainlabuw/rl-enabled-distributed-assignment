import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FilteredCOMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(FilteredCOMACritic, self).__init__()

        self.args = args
        self.m = args.m
        self.n = args.n
        self.M = args.env_args["M"]
        self.N = args.env_args["N"]
        self.L = args.env_args["L"]

        self.scheme = scheme
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, self.M + 1)

    def forward(self, batch, neighbors=None, top_agent_tasks=None, t=None):
        #if neighbors and top_agent_tasks are not provided, calculate them
        if neighbors is None or top_agent_tasks is None:
            bs = batch.batch_size
            max_t = batch.max_seq_length if t is None else 1

            total_beta = batch["beta"].float().sum(axis=-1)
            top_agent_tasks = th.topk(total_beta, k=self.M, dim=-1).indices

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
                neighbors[:,:,i,:] = top_N

        inputs = self._build_inputs(batch, neighbors, top_agent_tasks, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, neighbors, top_agent_tasks, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        #add beta for each agents neighbors
        beta = th.zeros((bs, max_t, self.n, self.N*self.M*self.L), device=batch.device)
        actions = th.zeros((bs, max_t, self.n, self.N*self.m), device=batch.device)
        power_states = th.zeros((bs, max_t, self.n, self.N), device=batch.device)
        prev_assigns = th.zeros((bs, max_t, self.n, self.N*self.M), device=batch.device)
        for i in range(self.n):
            neighbor_indices = np.indices(neighbors[:,ts, i].shape)
            top_agent_i_tasks = top_agent_tasks[:, ts, i]
            agent_i_neighbors = neighbors[:, ts, i]

            #beta of neighbors
            beta_neighbors = batch["beta"][:, ts][neighbor_indices[0], neighbor_indices[1], agent_i_neighbors, :, :]
            beta[:,:,i,:] = beta_neighbors[neighbor_indices[0],neighbor_indices[1],:,top_agent_i_tasks,:].view(bs, max_t, -1)
            # if i == 0: #NOTE: things are transposed? rows seems to be the tasks, columns the agents. not sure if this will matter as long as its consistent
            #     print(beta_neighbors_tasks[0,0,:,:,:].sum(axis=-1))
            #     for n in range(self.N):
            #         print(beta_neighbors_tasks[0,0,n,:,:].sum(axis=-1))

            #actions of neighbors
            relevant_actions = batch["actions_onehot"][:, ts][neighbor_indices[0], neighbor_indices[1], neighbors[:,:,i], :]
            actions[:,:,i,:] = relevant_actions.view(bs, max_t, -1)

            #power states of neighbors
            power_states[:,:,i,:] = batch["power_states"][:, ts][neighbor_indices[0], neighbor_indices[1], neighbors[:,:,i]]

            #previous assignments of neighbors
            neighbor_prev_assigns = batch["prev_assigns"][:, ts][neighbor_indices[0], neighbor_indices[1], neighbors[:,:,i]].to(th.int64)
            neighbor_prev_assigns_onehot = F.one_hot(neighbor_prev_assigns, num_classes=self.m)
            prev_assigns[:,:,i,:] = neighbor_prev_assigns_onehot[neighbor_indices[0], neighbor_indices[1], :, top_agent_i_tasks].view(bs, max_t, -1)



        inputs.append(beta)
        inputs.append(actions)
        inputs.append(power_states)
        inputs.append(prev_assigns)

        # observation
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

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
        # # get flattened size of state
        # for key in scheme.keys():
        #     if scheme[key].get("part_of_state", False):
        #         shape_contribution = 1
        #         for dim in scheme[key]["vshape"]:
        #             shape_contribution *= dim
        #         input_shape += shape_contribution
        #NOTE: hardcoding this.
        input_shape += self.N*self.M*self.L + self.N*self.m + self.N + self.N*self.M
        
        # observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # # actions (accounted for above)
        # input_shape += scheme["actions_onehot"]["vshape"][0] * self.N
        # last action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n
        return input_shape