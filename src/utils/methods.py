import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
from astropy import units as u

# Set the printing options to display more entries
np.set_printoptions(threshold=np.inf)

#~~~~~~~~~~~~~~~~~~~~HAAL UTILITIES~~~~~~~~~~~~~~
def build_time_interval_sequences(all_time_intervals, len_window):
    """
    Recursively constructs all possible time interval sequences from the set of all time intervals.

    Implements the logic behind BUILD_TIME_INTERVAL_SEQUENCES from the paper.
    """
    all_time_interval_sequences = []

    def build_time_interval_sequences_rec(all_time_intervals, time_interval_sequence, len_window):
        #Grab the most recent timestep from the end of the current sol
        if time_interval_sequence == []:
            most_recent_timestep = -1 #set it to -1 so that time intervals starting w 0 will be selected
        else:
            most_recent_timestep = time_interval_sequence[-1][-1]

        #When we have an time interval seq which ends at the last timestep, we're done
        #and can add it to the list
        if most_recent_timestep == (len_window-1):
            all_time_interval_sequences.append(tuple(time_interval_sequence))
        else:
            #Iterate through all of the time intervals, looking for ones that start where this one ended
            for time_interval in all_time_intervals:
                if most_recent_timestep == time_interval[0]-1:
                    build_time_interval_sequences_rec(all_time_intervals, time_interval_sequence + [time_interval], len_window)

    build_time_interval_sequences_rec(all_time_intervals, [], len_window)

    return all_time_interval_sequences

def generate_all_time_intervals(L):
    """
    Generates all possible time intervals from the next few timesteps.

    Implements GENERATE_ALL_TIME_INTERVALS from the paper.
    """
    all_time_intervals = []
    for i in range(L):
        for j in range(i,L):
            all_time_intervals.append((i,j))
        
    return all_time_intervals

#~~~~~~~~~~~~~~~~~~~~ EXPERIMENT UTILITIES ~~~~~~~~~~~~~~
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
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    pass_lens = []
    pass_bens = []
    task_assign_len = []
    for j in range(m):
        for i in range(n):
            pass_started = False
            task_assigned = False
            assign_len = 0
            pass_len = 0
            pass_ben = 0
            for k in range(T):
                this_pass_assign_lens = []
                if benefits[i,j,k] > 0:
                    if not pass_started:
                        pass_started = True
                    pass_len += 1
                    pass_ben += benefits[i,j,k]

                    if assigns is not None and assigns[k][i,j] == 1:
                        if not task_assigned: task_assigned = True
                        assign_len += 1
                    #If there are benefits and the task was previously assigned,
                    #but is no longer, end the streak
                    elif task_assigned:
                        task_assigned = False
                        this_pass_assign_lens.append(assign_len)
                        assign_len = 0

                elif pass_started and benefits[i,j,k] == 0:
                    if task_assigned:
                        this_pass_assign_lens.append(assign_len)
                    pass_started = False
                    task_assigned = False
                    for ass_len in this_pass_assign_lens:
                        task_assign_len.append(ass_len)
                    this_pass_assign_lens = []
                    pass_lens.append(pass_len)
                    pass_bens.append(pass_ben)
                    pass_len = 0
                    pass_ben = 0
                    assign_len = 0
    
    avg_pass_len = sum(pass_lens) / len(pass_lens)
    avg_pass_ben = sum(pass_bens) / len(pass_bens)

    if assigns is not None:
        avg_ass_len = sum(task_assign_len) / len(task_assign_len)
        return avg_pass_len, avg_pass_ben, avg_ass_len
    else:
        return avg_pass_len, avg_pass_ben
    
def propagate_sat_lat_lons(sat, T, dt):
    lats = []
    lons = []

    #Reset orbit to initial
    sat.orbit = sat.init_orbit
    for k in range(T):
        lats.append(sat.orbit.arglat.to_value(u.deg) % 360)
        lons.append((sat.orbit.L.to_value(u.deg)) % 360 - 15)
        sat.propagate_orbit(dt)
    
    return lats, lons

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