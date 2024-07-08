from main import experiment_run
import numpy as np
import matplotlib.pyplot as plt

from envs.HighPerformanceConstellationSim import HighPerformanceConstellationSim

def test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items=None, verbose=False):
    params = [
        'src/main.py',
        f'--config={alg_str}',
        f'--env-config={env_str}',
        'with',
        f'checkpoint_path={load_path}',
        'test_nepisode=1',
        'evaluate=True',
        'buffer_size=1',
        'runner=episode',
        'batch_size_run=1'
        ]
    if explicit_dict_items is None:
        explicit_dict_items = {
            'env_args': {'sat_prox_mat': sat_prox_mat,
                         "graphs": [1],} #placeholder
        }
    else:
        explicit_dict_items['env_args']['sat_prox_mat'] = sat_prox_mat
    
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]

    exp = experiment_run(params, explicit_dict_items, verbose=verbose)
    val = float(exp.result[1])
    actions = exp.result[0]
    assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in actions]

    if env_str == 'real_power_constellation_env':
        ps = exp.result[2]
        return assigns, val, ps
    else:
        return assigns, val

def test_classic_algorithms(alg_str, env_str, sat_prox_mat, explicit_dict_items=None, verbose=False):
    params = [
        'src/main.py',
        '--config=filtered_reda',
        f'--env-config={env_str}',
        'with',
        'test_nepisode=1',
        'evaluate=True',
        'jumpstart_evaluation_epsilon=1',
        f'jumpstart_action_selector=\"{alg_str}\"',
        'buffer_size=1'
        ]
    if explicit_dict_items is None:
        explicit_dict_items = {
            'env_args': {'sat_prox_mat': sat_prox_mat,
                        'graphs': [1], #placeholder
                        }
        }
    else:
        explicit_dict_items['env_args']['sat_prox_mat'] = sat_prox_mat

    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]

    exp = experiment_run(params, explicit_dict_items, verbose=verbose)
    val = float(exp.result[1])
    actions = exp.result[0]
    assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in actions]

    if env_str == 'real_power_constellation_env':
        ps = exp.result[2]
        return assigns, val, ps
    else:
        return assigns, val

def calc_handovers_generically(assignments, init_assign=None, benefit_info=None):
    """
    Calculate the number of handovers generically, without assuming that the handover penalty
    is the generic handover penalty, as opposed to calc_value_num_handovers above.
    """
    n = assignments[0].shape[0]
    m = assignments[0].shape[1]
    T = len(assignments)

    #If T_trans is provided, then use it, otherwise just set it to 
    try:
        T_trans = benefit_info.T_trans
    except AttributeError:
        T_trans = np.ones((m,m)) - np.eye(m)

    num_handovers = 0
    prev_assign = init_assign
    for k in range(T):
        if prev_assign is not None:
            new_assign = assignments[k]

            #iterate through agents
            for i in range(n):
                new_task_assigned = np.argmax(new_assign[i,:])
                prev_task_assigned = np.argmax(prev_assign[i,:])

                if prev_assign[i,new_task_assigned] == 0 and T_trans[prev_task_assigned,new_task_assigned] == 1:
                    num_handovers += 1
        
        prev_assign = assignments[k]

    return num_handovers

def calc_pct_conflicts(assignments):
    T = len(assignments)
    n = assignments[0].shape[0]
    m = assignments[0].shape[1]

    pct_conflicts = []
    for k in range(T):
        num_agents_w_conflicts = 0
        for i in range(n):
            assigned_task = np.argmax(assignments[k][i,:])
            if np.sum(assignments[k][:,assigned_task]) > 1:
                num_agents_w_conflicts += 1
        
        pct_conflicts.append(100*num_agents_w_conflicts / n)

    return pct_conflicts

def calc_pass_statistics(benefits, assigns=None):
    """
    Given a benefit array returns various statistics about the satellite passes over tasks.

    Note that we define a satellite pass as the length of time a satellite
    can obtain non-zero benefit for completing a given task.

    Specifically:
     - avg_pass_len: the average length of time a satellite is in view of a single task
            (even if the satellite is not assigned to the task)
     - avg_pass_ben: the average benefits that would be yielded for a satellite being
            assigned to a task for the whole time it is in view

    IF assigns is provided, then we also calculate:
     - avg_ass_len: the average length of time a satellite is assigned to the same task
            (only counted when the task the satellite is completing has nonzero benefit)
     - avg_ass_ben: the average benefits yielded by a satellite over the course of time
            it is assigned to the same task.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    pass_lens = []
    pass_bens = []
    task_assign_len = []
    task_assign_ben = []
    l = 0
    for j in range(m):
        for i in range(n):
            pass_started = False
            task_assigned = False
            assign_len = 0
            assign_ben = 0
            pass_len = 0
            pass_ben = 0
            this_pass_assign_lens = []
            this_pass_assign_bens = []
            for k in range(T):
                if benefits[i,j,k] > 0:
                    if not pass_started:
                        pass_started = True
                    pass_len += 1
                    pass_ben += benefits[i,j,k]

                    if assigns is not None and assigns[k][i,j] == 1:
                        if not task_assigned: 
                            task_assigned = True
                        assign_len += 1
                        assign_ben += benefits[i,j,k]
                    #If there are benefits and the task was previously assigned,
                    #but is no longer, end the streak
                    elif task_assigned:
                        task_assigned = False
                        this_pass_assign_lens.append(assign_len)
                        this_pass_assign_bens.append(assign_ben)
                        assign_len = 0
                        assign_ben = 0
                elif pass_started and benefits[i,j,k] == 0.0:
                    if task_assigned:
                        this_pass_assign_lens.append(assign_len)
                        this_pass_assign_bens.append(assign_ben)
                    pass_started = False
                    task_assigned = False
                    for ass_len in this_pass_assign_lens:
                        task_assign_len.append(ass_len)
                    for ass_ben in this_pass_assign_bens:
                        task_assign_ben.append(ass_ben)
                    this_pass_assign_lens = []
                    this_pass_assign_bens = []
                    pass_lens.append(pass_len)
                    pass_bens.append(pass_ben)
                    pass_len = 0
                    pass_ben = 0
                    assign_len = 0
                    assign_ben = 0
    
    avg_pass_len = sum(pass_lens) / len(pass_lens)
    avg_pass_ben = sum(pass_bens) / len(pass_bens)

    if assigns is not None:
        avg_ass_len = sum(task_assign_len) / len(task_assign_len)
        avg_ass_ben = sum(task_assign_ben) / len(task_assign_ben)
        return avg_pass_len, avg_pass_ben, avg_ass_len, avg_ass_ben
    else:
        return avg_pass_len, avg_pass_ben
    
def calc_meaningful_handovers(sat_prox_mat, assignments):
    """
    Calculate the number of handovers generically, without assuming that the handover penalty
    is the generic handover penalty, as opposed to calc_value_num_handovers above.
    """
    assignments = assignments[:-1] #last assign matrix is not real

    n = assignments[0].shape[0]
    m = assignments[0].shape[1]
    T = len(assignments)

    num_handovers = 0
    prev_assign = np.ones_like(assignments[0])
    for k in range(T):
        if prev_assign is not None:
            new_assign = assignments[k]

            #iterate through agents
            for i in range(n):
                new_task_assigned = np.argmax(new_assign[i,:])
                prev_task_assigned = np.argmax(prev_assign[i,:])

                if prev_assign[i,new_task_assigned] == 0 and sat_prox_mat[i, new_task_assigned, k] > 0:
                    num_handovers += 1
        
        prev_assign = assignments[k]

    return num_handovers

def convert_central_sol_to_assignment_mat(n, m, assignments):
    """
    Converts a list of column indices to an assignment matrix.
    (column indices are the output from scipy.optimize.linear_sum_assignment)

    i.e. for n=m=3, [1,2,0] -> [[0,1,0],[0,0,1],[1,0,0]]
    """
    assignment_mat = np.zeros((n, m), dtype="bool")
    for i, assignment in enumerate(assignments):
        assignment_mat[i, assignment] = 1

    return assignment_mat

def dictator_env_training():
    """
    Runs a training run of all the algorithms tested in the dictator environment.
    """
    def run_dict_experiment(alg_str):
        params = [
        'src/main.py',
        f'--config={alg_str}',
        f'--env-config=dictator_env',
        'with',
        'log_interval=50000',
        ]
        experiment_run(params, None, verbose=True)

    # REDA
    print("RUNNING REDA TRAINING ON DICTATOR ENV")
    run_dict_experiment('dictator_reda')
    
    # IQL
    print("RUNNING IQL TRAINING ON DICTATOR ENV")
    run_dict_experiment('dictator_iql')

    # IPPO
    print("RUNNING IPPO TRAINING ON DICTATOR ENV")
    run_dict_experiment('dictator_ippo')

    # COMA
    print("RUNNING COMA TRAINING ON DICTATOR ENV")
    run_dict_experiment('dictator_coma')

def constellation_env_test():
    """
    Tests the performance of models pretrained with REDA, IQL, IPPO, and COMA on the constellation environment.
    """
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 100
    L = 3
    lambda_ = 0.5
    env_str = 'real_power_constellation_env'

    reda_sat_ps = []
    iql_sat_ps = []
    haal_sat_ps = []
    alpha_beta_sat_ps = []
    ippo_sat_ps = []

    tot_iql_conflicts = []
    tot_ippo_conflicts = []

    reda_ass_len = []
    iql_ass_len = []
    ippo_ass_len = []
    haal_ass_len = []
    alpha_beta_ass_len = []

    reda_vals = []
    ippo_vals = []
    iql_vals = []
    haal_vals = []
    alpha_beta_vals = []

    num_tests = 1
    for _ in range(num_tests):
        print(f"Testing algs on random constellation {_+1}/{num_tests}")
        const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
        sat_prox_mat = const.get_proximities_for_random_tasks(m)
        explicit_dict_items = {
            'env_args': {'sat_prox_mat': sat_prox_mat,
                        'graphs': const.graphs,
                        'T': T,
                        'lambda_': lambda_,
                        'L': L,
                        }
        }

        # REDA
        alg_str = 'filtered_reda'
        load_path = './pretrained_models/reda_constellation_env'
        reda_assigns, reda_val, reda_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=True)
        reda_sat_ps.append(np.sum(np.where(reda_ps > 0, 1, 0)) / n)

        _, _, reda_al, _ = calc_pass_statistics(sat_prox_mat, reda_assigns)
        reda_ass_len.append(reda_al)
        reda_vals.append(reda_val)

        # IQL
        alg_str = 'filtered_iql'
        load_path = './pretrained_models/iql_constellation_env'
        iql_assigns, iql_val, iql_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

        iql_sat_ps.append(np.sum(np.where(iql_ps > 0, 1, 0)) / n)

        tot_iql_conflicts.extend(calc_pct_conflicts(iql_assigns))

        _, _, iql_al, _ = calc_pass_statistics(sat_prox_mat, iql_assigns)
        iql_ass_len.append(iql_al)
        iql_vals.append(iql_val)

        # IPPO
        alg_str = 'filtered_ippo'
        load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_ippo_seed194208545_2024-05-20 21:35:11.427806'
        ippo_assigns, ippo_val, ippo_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

        tot_ippo_conflicts.extend(calc_pct_conflicts(ippo_assigns))

        _, _, ippo_al, _ = calc_pass_statistics(sat_prox_mat, ippo_assigns)
        ippo_ass_len.append(ippo_al)
        ippo_sat_ps.append(np.sum(np.where(ippo_ps > 0, 1, 0)) / n)
        ippo_vals.append(ippo_val)

        # HAAL
        haal_assigns, haal_val, haal_ps = test_classic_algorithms('haal_selector', env_str, sat_prox_mat, verbose=False)
        haal_sat_ps.append(np.sum(np.where(haal_ps > 0, 1, 0)) / n)

        _, _, haal_al, haal_ab = calc_pass_statistics(sat_prox_mat, haal_assigns)
        haal_ass_len.append(haal_al)
        haal_vals.append(haal_val)

        # \alpha(\beta)
        alpha_beta_assigns, alpha_beta_val, alpha_beta_ps = test_classic_algorithms('alpha_beta_selector', env_str, sat_prox_mat)
        alpha_beta_sat_ps.append(np.sum(np.where(alpha_beta_ps > 0, 1, 0)) / n)

        _, _, alpha_beta_al, _ = calc_pass_statistics(sat_prox_mat, alpha_beta_assigns)
        alpha_beta_ass_len.append(alpha_beta_al)
        alpha_beta_vals.append(alpha_beta_val)

    iql_mean_conflicts = np.mean(np.array(tot_iql_conflicts))
    ippo_mean_conflicts = np.mean(np.array(tot_ippo_conflicts))

    iql_std_conflicts = np.std(np.array(tot_iql_conflicts))
    ippo_std_conflicts = np.std(np.array(tot_ippo_conflicts))

    reda_mean_ps = np.mean(np.array(reda_sat_ps))
    iql_mean_ps = np.mean(np.array(iql_sat_ps))
    haal_mean_ps = np.mean(np.array(haal_sat_ps))
    alpha_beta_mean_ps = np.mean(np.array(alpha_beta_sat_ps))
    ippo_mean_ps = np.mean(np.array(ippo_sat_ps))

    power_means = np.array([alpha_beta_mean_ps, haal_mean_ps, ippo_mean_ps, iql_mean_ps, reda_mean_ps])

    reda_std_ps = np.std(np.array(reda_sat_ps))
    iql_std_ps = np.std(np.array(iql_sat_ps))
    haal_std_ps = np.std(np.array(haal_sat_ps))
    alpha_beta_std_ps = np.std(np.array(alpha_beta_sat_ps))
    ippo_std_ps = np.std(np.array(ippo_sat_ps))

    power_stds = np.array([alpha_beta_std_ps, haal_std_ps, ippo_std_ps, iql_std_ps, reda_std_ps])

    reda_mean_al = np.mean(np.array(reda_ass_len))
    iql_mean_al = np.mean(np.array(iql_ass_len))
    haal_mean_al = np.mean(np.array(haal_ass_len))
    alpha_beta_mean_al = np.mean(np.array(alpha_beta_ass_len))
    ippo_mean_al = np.mean(np.array(ippo_ass_len))

    al_means = np.array([alpha_beta_mean_al, haal_mean_al, ippo_mean_al, iql_mean_al, reda_mean_al])

    reda_std_al = np.std(np.array(reda_ass_len))
    iql_std_al = np.std(np.array(iql_ass_len))
    haal_std_al = np.std(np.array(haal_ass_len))
    alpha_beta_std_al = np.std(np.array(alpha_beta_ass_len))
    ippo_std_al = np.std(np.array(ippo_ass_len))

    al_stds = np.array([alpha_beta_std_al, haal_std_al, ippo_std_al, iql_std_al, reda_std_al])

    # Define the data
    categories = ['% Sats with \n charge at k=100', '% Sats with conflict\nfree assignments', 'Avg. # Steps Assigned\n to same task (normalized)']
    data = np.array([100*power_means, [99.9, 99.9, 100-ippo_mean_conflicts, 100-iql_mean_conflicts, 99.9], 100*al_means/max(al_means)])
    error = np.array([100*power_stds, [0, 0, iql_std_conflicts, ippo_std_conflicts, 0], 100*al_stds/max(al_means)])

    # Set the width of each bar and the spacing between categories
    bar_width = 0.25
    category_spacing = np.arange(len(categories))*1.5

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10.3, 5))

    labels = [r"$\alpha(\hat{\beta}(s))$", 'HAAL', 'IPPO', 'IQL', 'REDA']
    colors = ['gray', 'green', 'red', 'blue', 'purple']
    # Plot the bars and error bars for each category
    for i in range(len(labels)):
        ax.bar(category_spacing + i * bar_width, data[:, i], bar_width, 
            yerr=error[:, i], capsize=5, label=labels[i], color=colors[i])

    # Set the x-axis tick positions and labels
    ax.set_xticks(category_spacing + bar_width * (len(labels) - 1) / 2)
    ax.set_xticklabels(categories)

    # Add a legend
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='center left')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))

    means = [np.mean(alpha_beta_vals), np.mean(haal_vals), np.mean(ippo_vals), np.mean(iql_vals), np.mean(reda_vals)]
    std_devs = [np.std(alpha_beta_vals), np.std(haal_vals), np.std(ippo_vals), np.std(iql_vals), np.std(reda_vals)]

    ax.bar(labels, means, yerr=std_devs, capsize=5)

    ax.set_ylabel('Mean Value')
    ax.set_title('Comparison of Algorithms')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # # Train REDA, IQL, IPPO, COMA on dictator environment from scratch
    # dictator_env_training()

    # Test pretrained algorithms on constellation environment
    constellation_env_test()