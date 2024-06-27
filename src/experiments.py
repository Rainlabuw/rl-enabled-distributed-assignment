from astropy import units as u
from main import experiment_run
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import sys
import copy
import pickle
sys.path.append('/Users/joshholder/code/satellite-constellation')

from controllers.basic_controller import BasicMAC
from envs.mock_constellation_env import generate_benefits_over_time, MockConstellationEnv
from envs.power_constellation_env import PowerConstellationEnv
from envs.real_constellation_env import RealConstellationEnv
from envs.real_power_constellation_env import RealPowerConstellationEnv

from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_randomly import solve_randomly
from algorithms.solve_greedily import solve_greedily
from algorithms.solve_wout_handover import solve_wout_handover
from haal_experiments.simple_assign_env import SimpleAssignEnv
from common.methods import *

from envs.HighPerformanceConstellationSim import HighPerformanceConstellationSim
from envs.StarlinkSim import StarlinkSim
from constellation_sim.constellation_generators import get_prox_mat_and_graphs_random_tasks

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

def mock_constellation_test():
    n = 10
    m = 10
    T = 15
    L = 3
    lambda_ = 0.5

    np.random.seed(44)
    sat_prox_mat = generate_benefits_over_time(n, m, T, 3, 6)
    
    #EVALUATE VDN
    print('Evaluating VDN')
    vdn_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_mock_const'
    params = [
        'src/main.py',
        '--config=vdn',
        '--env-config=mock_constellation_env',
        'with',
        f'checkpoint_path={vdn_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=False)
    vdn_val = vdn_exp.result[1]
    vdm = vdn_exp.result[0]
    vdn_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdm]
    
    #EVALUATE AUCTION VDN
    print('Evaluating Auction VDN')
    vdn_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_sap_mock_const'
    params = [
        'src/main.py',
        '--config=vdn_sap',
        '--env-config=mock_constellation_env',
        'with',
        f'checkpoint_path={vdn_sap_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat}
    }
    
    vdn_sap_exp = experiment_run(params, explicit_dict_items, verbose=False)
    vdn_sap_val = vdn_sap_exp.result[1]
    vdn_sap_actions = vdn_sap_exp.result[0]
    vdn_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdn_sap_actions]
    
    print('VDN:', vdn_val)
    print('VDN SAP:', vdn_sap_val)
    #EVALUATE CLASSIC ALGORITHMS
    env = MockConstellationEnv(n, m, T, L, lambda_, sat_prox_mat=sat_prox_mat)
    haal_assigns, haal_val = solve_w_haal(env, L, verbose=False)
    print('HAAL:', haal_val)

    env.reset()
    random_assigns, random_val = solve_randomly(env)
    print('Random:', random_val)

    env.reset()
    greedy_assigns, greedy_val = solve_greedily(env)
    print('Greedy:', greedy_val)

    env.reset()
    nha_assigns, nha_val = solve_wout_handover(env)
    print('Without Handover:', nha_val)

    values = [vdn_val, vdn_sap_val, haal_val, random_val, greedy_val, nha_val]
    handovers = [calc_handovers_generically(a) for a in [vdn_assigns, vdn_sap_assigns, haal_assigns, random_assigns, greedy_assigns, nha_assigns]]
    
    alg_names = ['VDN', 'VDN SAP', 'HAAL', 'Random', 'Greedy', 'Without Handover']
    plt.bar(alg_names, values)
    plt.show()

    plt.bar(alg_names, handovers)
    plt.show()

def power_constellation_test():
    n = 10
    m = 10
    T = 15
    L = 3
    lambda_ = 0.5

    np.random.seed(44)
    sat_prox_mat = generate_benefits_over_time(n, m, T, 3, 6)
    
    #EVALUATE VDN
    print('Evaluating VDN')
    vdn_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_power_const'
    params = [
        'src/main.py',
        '--config=vdn',
        '--env-config=power_constellation_env',
        'with',
        f'checkpoint_path={vdn_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'T': T}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=True)
    vdn_val = vdn_exp.result[1]
    vdm = vdn_exp.result[0]
    vdn_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdm]
    
    #EVALUATE AUCTION VDN
    print('Evaluating Auction VDN')
    vdn_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_sap_power_const'
    params = [
        'src/main.py',
        '--config=vdn_sap',
        '--env-config=power_constellation_env',
        'with',
        f'checkpoint_path={vdn_sap_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'T': T}
    }
    
    vdn_sap_exp = experiment_run(params, explicit_dict_items, verbose=True)
    vdn_sap_val = vdn_sap_exp.result[1]
    vdn_sap_actions = vdn_sap_exp.result[0]
    vdn_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdn_sap_actions]
    
    print('VDN:', vdn_val)
    print('VDN SAP:', vdn_sap_val)
    #EVALUATE CLASSIC ALGORITHMS
    env = PowerConstellationEnv(n, m, T, L, lambda_, sat_prox_mat=sat_prox_mat)
    haal_assigns, haal_val = solve_w_haal(env, L, verbose=False)
    print('HAAL:', haal_val)

    env.reset()
    random_assigns, random_val = solve_randomly(env)
    print('Random:', random_val)

    env.reset()
    greedy_assigns, greedy_val = solve_greedily(env)
    print('Greedy:', greedy_val)

    env.reset()
    nha_assigns, nha_val = solve_wout_handover(env)
    print('Without Handover:', nha_val)

    values = [vdn_val, vdn_sap_val, haal_val, random_val, greedy_val, nha_val]
    handovers = [calc_handovers_generically(a) for a in [vdn_assigns, vdn_sap_assigns, haal_assigns, random_assigns, greedy_assigns, nha_assigns]]
    
    alg_names = ['VDN', 'VDN SAP', 'HAAL', 'Random', 'Greedy', 'Without Handover']
    plt.bar(alg_names, values)
    plt.xlabel('Algorithm')
    plt.ylabel('Value')
    plt.show()

    plt.bar(alg_names, handovers)
    plt.xlabel('Algorithm')
    plt.ylabel('Handovers')
    plt.show()

def neighborhood_benefits_test():
    sat_prox_mat = np.zeros((4,4,2))
    sat_prox_mat[:,:,0] = np.array([[5, 0, 0, 1],
                                    [2, 0, 0, 0],
                                    [3, 1, 4, 2],
                                    [1, 3, 0, 10]])
    sat_prox_mat[:,:,1] = np.ones((4,4))
    
    params = [
        'src/main.py',
        '--config=iql_sap',
        '--env-config=real_constellation_env',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': 1, #so its not None
                     'M': 2,
                     'N': 2,
                     'L': 2,
                     'm': 4,
                     'T': 1}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=True)

def real_power_constellation_test():
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes * num_sats_per_plane
    m = 150
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
    sat_prox_mat = const.get_proximities_for_random_tasks(m)

    basic_params = [
        'src/main.py',
        '--config=iql_sap',
        '--env-config=real_power_constellation_env',
        'with',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }

    #~~~~~~~~~ EVALUATE IQL SAP ~~~~~~~~~~
    iql_sap_params = copy.copy(basic_params)

    iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/flat_iql_real_power'
    iql_sap_params.append(f'checkpoint_path={iql_sap_model_path}')

    iql_sap_exp = experiment_run(iql_sap_params, explicit_dict_items, verbose=False)
    iql_sap_val = float(iql_sap_exp.result[1])
    iql_sap_actions = iql_sap_exp.result[0]
    iql_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in iql_sap_actions]

    #~~~~~~~~~ EVALUATE HAA ~~~~~~~~~~
    print("EVALUATING HAA")
    haa_params = copy.copy(basic_params)

    # haa_params.append(f'checkpoint_path={iql_sap_model_path}')
    haa_params.append('jumpstart_evaluation_epsilon=1')
    haa_params.append('jumpstart_action_selector=\"haa_selector\"')

    haa_exp = experiment_run(haa_params, explicit_dict_items, verbose=True)
    haa_val = float(haa_exp.result[1])
    haa_actions = haa_exp.result[0]
    haa_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haa_actions]

    #~~~~~~~~~ EVALUATE HAAL ~~~~~~~~~~
    print("EVALUATING HAAL")
    haal_params = copy.copy(basic_params)
    haal_params.append('jumpstart_evaluation_epsilon=1')
    haal_params.append('jumpstart_action_selector=\"haal_selector\"')

    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }

    haal_exp = experiment_run(haal_params, explicit_dict_items, verbose=True)
    haal_val = float(haal_exp.result[1])
    haal_actions = haal_exp.result[0]
    haal_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haal_actions]
    

    print('IQL SAP:', iql_sap_val)
    print('HAA:', haa_val)
    print('HAAL:', haal_val)

def haal_test():
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes * num_sats_per_plane
    m = 150
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
    sat_prox_mat = const.get_proximities_for_random_tasks(m)

    params = [
        'src/main.py',
        '--config=iql_sap_custom_cnn',
        '--env-config=real_power_constellation_env',
        'with',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        'jumpstart_evaluation_epsilon=1',
        'jumpstart_action_selector=\"haal_selector\"'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }

    haal_exp = experiment_run(params, explicit_dict_items, verbose=True)
    haal_val = float(haal_exp.result[1])
    haal_actions = haal_exp.result[0]
    haal_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haal_actions]
    print('HAAL:', haal_val)

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
        
        pct_conflicts.append(num_agents_w_conflicts / n)

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

def real_power_compare():
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
    haa_sat_ps = []
    ippo_sat_ps = []
    haa_heur_sat_ps = []
    haal_heur_sat_ps = []

    tot_iql_conflicts = []
    tot_ippo_conflicts = []

    reda_ass_len = []
    iql_ass_len = []
    ippo_ass_len = []
    haal_ass_len = []
    haa_ass_len = []
    haa_heur_ass_len = []
    # haal_heur_ass_len = [] #shown to be bad

    # total_haal_heur_val = 0 #shown to be bad
    total_haa_heur_val = 0
    total_reda_val = 0
    total_ippo_val = 0
    total_iql_val = 0
    total_haal_val = 0
    total_haa_val = 0

    num_tests = 5
    for _ in range(num_tests):
        print(_)
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
        load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_reda_seed952807856_2024-05-15 21:26:29.301905'
        reda_assigns, reda_val, reda_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

        reda_sat_ps.append(np.sum(np.where(reda_ps > 0, 1, 0)) / 324)

        _, _, reda_al, reda_ab = calc_pass_statistics(sat_prox_mat, reda_assigns)
        # nh = calc_meaningful_handovers(sat_prox_mat, reda_assigns)
        # print("REDA ASS LEN", reda_al, reda_ab, nh)
        # print("reda % still charged", np.sum(np.where(reda_ps > 0, 1, 0)) / 324)
        reda_ass_len.append(reda_al)
        total_reda_val += reda_val

        # IQL
        alg_str = 'filtered_iql'
        load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_iql_seed814515160_2024-05-17 01:23:22.125853'
        iql_assigns, iql_val, iql_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

        iql_sat_ps.append(np.sum(np.where(iql_ps > 0, 1, 0)) / 324)

        tot_iql_conflicts.extend(calc_pct_conflicts(iql_assigns))

        _, _, iql_al, iql_ab = calc_pass_statistics(sat_prox_mat, iql_assigns)
        # print("IQL ASS LEN", iql_al, iql_ab)
        # print(np.sum(np.where(iql_ps > 0, 1, 0)) / 324)
        # print(np.mean(calc_pct_conflicts(iql_assigns)))
        iql_ass_len.append(iql_al)
        total_iql_val += iql_val

        # IPPO
        alg_str = 'filtered_ippo'
        load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_ippo_seed194208545_2024-05-20 21:35:11.427806'
        ippo_assigns, ippo_val, ippo_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

        tot_ippo_conflicts.extend(calc_pct_conflicts(ippo_assigns))

        _, _, ippo_al, ippo_ab = calc_pass_statistics(sat_prox_mat, ippo_assigns)
        # print("IPPO ASS LEN", ippo_al, ippo_ab)
        # print(np.sum(np.where(ippo_ps > 0, 1, 0)) / 324)
        # print(np.mean(calc_pct_conflicts(ippo_assigns)))
        ippo_ass_len.append(ippo_al)
        ippo_sat_ps.append(np.sum(np.where(ippo_ps > 0, 1, 0)) / 324)
        total_ippo_val += ippo_val


        haal_assigns, haal_val, haal_ps = test_classic_algorithms('haal_selector', env_str, sat_prox_mat, verbose=False)
        haal_sat_ps.append(np.sum(np.where(haal_ps > 0, 1, 0)) / 324)

        _, _, haal_al, haal_ab = calc_pass_statistics(sat_prox_mat, haal_assigns)
        # print("HAAL ASS LEN", haal_al, haal_ab)
        # print(np.sum(np.where(haal_ps > 0, 1, 0)) / 324)
        haal_ass_len.append(haal_al)
        total_haal_val += haal_val


        haa_assigns, haa_val, haa_ps = test_classic_algorithms('haa_selector', env_str, sat_prox_mat)
        haa_sat_ps.append(np.sum(np.where(haa_ps > 0, 1, 0)) / 324)

        _, _, haa_al, haa_ab = calc_pass_statistics(sat_prox_mat, haa_assigns)
        # print("HAA ASS LEN", haa_al, haa_ab)
        # print(np.sum(np.where(haa_ps > 0, 1, 0)) / 324)
        haa_ass_len.append(haa_al)
        total_haa_val += haa_val


        haa_heur_assigns, haa_heur_val, haa_heur_ps = test_classic_algorithms('haa_power_heuristic_selector', env_str, sat_prox_mat, explicit_dict_items=explicit_dict_items)
        haa_heur_sat_ps.append(np.sum(np.where(haa_heur_ps > 0, 1, 0)) / 324)

        _, _, haa_heur_al, haa_heur_ab = calc_pass_statistics(sat_prox_mat, haa_heur_assigns)
        # nh = calc_meaningful_handovers(sat_prox_mat, haa_heur_assigns)
        # print("Heuristic AL AB", haa_heur_al, haa_heur_ab, nh)
        haa_heur_ass_len.append(haa_heur_al)
        total_haa_heur_val += haa_heur_val

        # # HAAL HEURISTIC ~~~~~~~~~~~~~~
        # haal_heur_assigns, haal_heur_val, haal_heur_ps = test_classic_algorithms('haal_power_heuristic_selector', env_str, sat_prox_mat)
        # haal_heur_sat_ps.append(np.sum(np.where(haal_heur_ps > 0, 1, 0)) / 324)

        # _, _, haal_heur_al, haal_heur_ab = calc_pass_statistics(sat_prox_mat, haal_heur_assigns)
        # # nh = calc_meaningful_handovers(sat_prox_mat, haa_heur_assigns)
        # # print("Heuristic AL AB", haa_heur_al, haa_heur_ab, nh)
        # haal_heur_ass_len.append(haal_heur_al)
        # total_haal_heur_val += haal_heur_val

    iql_mean_conflicts = np.mean(np.array(tot_iql_conflicts))
    ippo_mean_conflicts = np.mean(np.array(tot_ippo_conflicts))

    iql_std_conflicts = np.std(np.array(tot_iql_conflicts))
    ippo_std_conflicts = np.std(np.array(tot_ippo_conflicts))

    print(iql_mean_conflicts, ippo_mean_conflicts)
    print(iql_std_conflicts, ippo_std_conflicts)

    reda_mean_ps = np.mean(np.array(reda_sat_ps))
    iql_mean_ps = np.mean(np.array(iql_sat_ps))
    haal_mean_ps = np.mean(np.array(haal_sat_ps))
    haa_mean_ps = np.mean(np.array(haa_sat_ps))
    ippo_mean_ps = np.mean(np.array(ippo_sat_ps))
    haa_heur_mean_ps = np.mean(np.array(haa_heur_sat_ps))
    # haal_heur_mean_ps = np.mean(np.array(haal_heur_sat_ps))

    power_means = np.array([reda_mean_ps, iql_mean_ps, ippo_mean_ps, haal_mean_ps, haa_mean_ps])

    reda_std_ps = np.std(np.array(reda_sat_ps))
    iql_std_ps = np.std(np.array(iql_sat_ps))
    haal_std_ps = np.std(np.array(haal_sat_ps))
    haa_std_ps = np.std(np.array(haa_sat_ps))
    ippo_std_ps = np.std(np.array(ippo_sat_ps))
    haa_heur_std_ps = np.std(np.array(haa_heur_sat_ps))
    # haal_heur_std_ps = np.std(np.array(haal_heur_sat_ps))

    power_stds = np.array([reda_std_ps, iql_std_ps, ippo_std_ps, haal_std_ps, haa_std_ps])
    print("POWER")
    print(power_means)
    print(power_stds)


    reda_mean_al = np.mean(np.array(reda_ass_len))
    iql_mean_al = np.mean(np.array(iql_ass_len))
    haal_mean_al = np.mean(np.array(haal_ass_len))
    haa_mean_al = np.mean(np.array(haa_ass_len))
    ippo_mean_al = np.mean(np.array(ippo_ass_len))
    haa_heur_mean_al = np.mean(np.array(haa_heur_ass_len))
    # haal_heur_mean_al = np.mean(np.array(haal_heur_ass_len))

    al_means = np.array([reda_mean_al, iql_mean_al, ippo_mean_al, haal_mean_al, haa_mean_al])

    reda_std_al = np.std(np.array(reda_ass_len))
    iql_std_al = np.std(np.array(iql_ass_len))
    haal_std_al = np.std(np.array(haal_ass_len))
    haa_std_al = np.std(np.array(haa_ass_len))
    ippo_std_al = np.std(np.array(ippo_ass_len))
    haa_heur_std_al = np.std(np.array(haa_heur_ass_len))
    # haal_heur_std_al = np.std(np.array(haal_heur_ass_len))

    al_stds = np.array([reda_std_al, iql_std_al, ippo_std_al, haal_std_al, haa_std_al])
    print("AL")
    print(al_means)
    print(al_stds)


    # print("HAAL HEUR VAL", total_haal_heur_val / num_tests)
    print("HAA HEUR VAL", total_haa_heur_val / num_tests)
    print("REDA VAL", total_reda_val / num_tests)
    print("HAA VAL", total_haa_val / num_tests)
    print("HAAL VAL", total_haal_val / num_tests)
    print("IQL VAL", total_iql_val / num_tests)
    print("IPPO VAL", total_ippo_val / num_tests)

    iql_mean_conflicts = 80.39115022613372
    ippo_mean_conflicts = 55.12101210121012
    iql_std_conflicts = 5.07837616102471
    ippo_std_conflicts = 5.423952404031624

    power_means = np.array([0.96111111, 0.91790123, 0.9654321, 0.001, 0.001])[::-1]*100
    power_stds = np.array([0.0081892, 0.01346755, 0.02329378, 0.0, 0.0])[::-1]*100

    al_means = np.array([1.7876856,  1.72106877, 1.78546133, 1.9654943, 2.32960784])[::-1]
    al_stds = np.array([0.03545114, 0.02069343, 0.0285523,  0.02834858, 0.04133362])[::-1]

    # Define the data
    categories = ['% Sats with \n charge at k=100', '% Sats with conflict\nfree assignments', 'Avg. # Steps Assigned\n to same task (normalized)']
    data = np.array([power_means, [99.9, 99.9, 100-ippo_mean_conflicts, 100-iql_mean_conflicts, 99.9], 100*al_means/max(al_means)])  # Replace with your data
    error = np.array([power_stds, [0, 0, iql_std_conflicts, ippo_std_conflicts, 0], 100*al_stds/max(al_means)])  # Replace with your actual standard deviations

    # Set the width of each bar and the spacing between categories
    bar_width = 0.25
    category_spacing = np.arange(len(categories))*1.5

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10.3, 5))

    labels = ["REDA", "IQL", "IPPO", "HAAL", r"$\alpha(\hat{\beta}(s))$"]
    labels.reverse()
    colors = ['purple', 'blue', 'red', 'green', 'gray']
    colors.reverse()
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

    # Add labels and title

    plt.tight_layout()
    plt.savefig('qual_perf_compare.pdf')
    plt.show()

def large_real_test():
    env_str = "real_constellation_env"
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 100
    L = 3
    lambda_ = 0.5
    init_assign = np.zeros((n,m))
    init_assign[:n, :n] = np.eye(n)

    N = 10
    M = 10

    total_haal = []
    total_haa = []
    total_reda_val = []
    total_greedy_val = []

    haal_nhs = []
    haa_nhs = []
    reda_nhs = []
    greedy_nhs = []

    haal_als = []
    haa_als = []
    reda_als = []
    greedy_als = []

    num_tests = 5
    for _ in range(num_tests):
        print("TEST ",_)
        env = RealConstellationEnv(num_planes, num_sats_per_plane, m, T, N, M, L, lambda_,
                                        task_prios=np.ones(m))
        env.reset()

        old_env = SimpleAssignEnv(env.sat_prox_mat, init_assign, L, lambda_)
        old_env.reset()
        greedy_assigns, greedy_val = solve_greedily(old_env)
        total_greedy_val.append(greedy_val)
        greedy_nh = calc_meaningful_handovers(env.sat_prox_mat, greedy_assigns)
        greedy_nhs.append(greedy_nh)
        _, _, greedy_al, _ = calc_pass_statistics(env.sat_prox_mat, greedy_assigns)
        greedy_als.append(greedy_al)

        haal_assigns, haal_val = test_classic_algorithms('haal_selector', env_str, env.sat_prox_mat, verbose=False)
        total_haal.append(haal_val)
        haal_nh = calc_meaningful_handovers(env.sat_prox_mat, haal_assigns)
        haal_nhs.append(haal_nh)
        _, _, haal_al, _ = calc_pass_statistics(env.sat_prox_mat, haal_assigns)
        haal_als.append(haal_al)

        haa_assigns, haa_val = test_classic_algorithms('haa_selector', env_str, env.sat_prox_mat)
        total_haa.append(haa_val)
        haa_nh = calc_meaningful_handovers(env.sat_prox_mat, haa_assigns)
        haa_nhs.append(haa_nh)
        _, _, haa_al, _ = calc_pass_statistics(env.sat_prox_mat, haa_assigns)
        haa_als.append(haa_al)

        load_path = '/Users/joshholder/code/marl_sap/results/models/large_real_no_power'
        reda_assigns, reda_val = test_rl_model('filtered_reda', env_str, load_path, env.sat_prox_mat)
        total_reda_val.append(reda_val)
        reda_nh = calc_meaningful_handovers(env.sat_prox_mat, reda_assigns)
        reda_nhs.append(reda_nh)
        _, _, reda_al, _ = calc_pass_statistics(env.sat_prox_mat, reda_assigns)
        reda_als.append(reda_al)

    mean_haal_val = np.sum(total_haal) / num_tests
    mean_haa_val = np.sum(total_haa) / num_tests
    mean_reda_val = np.sum(total_reda_val) / num_tests
    mean_greedy_val = np.sum(total_greedy_val) / num_tests

    std_haal_val = np.std(total_haal)
    std_haa_val = np.std(total_haa)
    std_reda_val = np.std(total_reda_val)
    std_greedy_val = np.std(total_greedy_val)

    mean_haal_nh = np.sum(haal_nhs) / num_tests
    mean_haa_nh = np.sum(haa_nhs) / num_tests
    mean_reda_nh = np.sum(reda_nhs) / num_tests
    mean_greedy_nh = np.sum(greedy_nhs) / num_tests

    std_haal_nh = np.std(haal_nhs)
    std_haa_nh = np.std(haa_nhs)
    std_reda_nh = np.std(reda_nhs)
    std_greedy_nh = np.std(greedy_nhs)

    mean_haal_al = np.sum(haal_als) / num_tests
    mean_haa_al = np.sum(haa_als) / num_tests
    mean_reda_al = np.sum(reda_als) / num_tests
    mean_greedy_al = np.sum(greedy_als) / num_tests

    std_haal_al = np.std(haal_als)
    std_haa_al = np.std(haa_als)
    std_reda_al = np.std(reda_als)
    std_greedy_al = np.std(greedy_als)

    print('HAAL:', mean_haal_val)
    print('HAA:', mean_haa_val)
    print('REDA:', mean_reda_val)
    print('GREEDY:', mean_greedy_val)

    print('HAAL std:', std_haal_val)
    print('HAA std:', std_haa_val)
    print('REDA std:', std_reda_val)
    print('GREEDY std:', std_greedy_val)

    print('HAAL nh:', mean_haal_nh)
    print('HAA nh:', mean_haa_nh)
    print('REDA nh:', mean_reda_nh)
    print('GREEDY nh:', mean_greedy_nh)

    print('HAAL nh std:', std_haal_nh)
    print('HAA nh std:', std_haa_nh)
    print('REDA nh std:', std_reda_nh)
    print('GREEDY nh std:', std_greedy_nh)

    print('HAAL al:', mean_haal_al)
    print('HAA al:', mean_haa_al)
    print('REDA al:', mean_reda_al)
    print('GREEDY al:', mean_greedy_al)

    print('HAAL al std:', std_haal_al)
    print('HAA al std:', std_haa_al)
    print('REDA al std:', std_reda_al)
    print('GREEDY al std:', std_greedy_al)

    score_data = [mean_haa_val, mean_greedy_val, mean_haal_val, mean_reda_val]
    score_error = [std_haa_val, std_greedy_val, std_haal_val, std_reda_val]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(1,2, figsize=(10,5))

    labels = [r"$\alpha(\hat{\beta}(s))$", "GA", "HAAL", "REDA"]
    colors = ['gray', 'blue', 'green', 'purple']
    # Plot the bars and error bars for each category
    axes[0].bar(labels, score_data, 
        yerr=score_error, capsize=5, color=colors)

    # nh_data = [mean_haa_nh, mean_greedy_nh, mean_haal_nh, mean_reda_nh]
    # nh_error = [std_haa_nh, std_greedy_nh, std_haal_nh, std_reda_nh]

    # axes[1].bar(labels, nh_data, 
    #     yerr=nh_error, capsize=5, color=colors)
    
    al_data = [mean_haa_al, mean_greedy_al, mean_haal_al, mean_reda_al]
    al_error = [std_haa_al, std_greedy_al, std_haal_al, std_reda_al]

    axes[1].bar(labels, al_data, 
        yerr=al_error, capsize=5, color=colors)

    # Add labels and title
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Value')

    axes[1].set_xlabel('Avg. # Steps Assigned\n to same meaningful task')
    axes[1].set_ylabel('# Steps')

    plt.tight_layout()
    plt.savefig('compare_reda_on_nopower.pdf')
    plt.show()

def large_real_power_test():
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 90
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    env_str = 'real_power_constellation_env'

    total_haal = 0
    total_haa = 0
    total_real_power = 0
    total_reda = 0

    num_tests = 5
    for _ in range(num_tests):
        print("TEST ",_)
        env = RealPowerConstellationEnv(num_planes, num_sats_per_plane, m, T, N, M, L, lambda_)
        env.reset()

        haa_heur_assigns, haa_heur_val = test_classic_algorithms('haa_power_heuristic_selector', env_str, env.sat_prox_mat)
        total_real_power += haa_heur_val
        
        haal_assigns, haal_val = test_classic_algorithms('haal_selector', env_str, env.sat_prox_mat)
        total_haal += haal_val

        haa_assigns, haa_val = test_classic_algorithms('haa_selector', env_str, env.sat_prox_mat)
        total_haa += haa_val

        load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_reda_seed952807856_2024-05-15 21:26:29.301905'
        reda_assigns, reda_val = test_rl_model('filtered_reda', env_str, load_path, env.sat_prox_mat)
        total_reda += reda_val
    
    print('HAAL:', total_haal / num_tests)
    print('HAA:', total_haa / num_tests)
    print('Real Power:', total_real_power / num_tests)
    print('REDA:', total_reda / num_tests)

if __name__ == "__main__":
    determine_good_M_N()