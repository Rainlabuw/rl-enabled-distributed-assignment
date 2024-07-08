from astropy import units as u
import numpy as np
import networkx as nx
import time
from copy import deepcopy

from collections import defaultdict

import sys

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
from poliastro.core.events import line_of_sight

class HighPerformanceConstellationSim(object):
    def __init__(self, num_planes, num_sats_per_plane, T=None, dt=63.76469*u.second, inc=58, altitude=550, isl_dist=np.inf, dtype=np.float64, use_graphs=False) -> None:
        self.inc = inc*u.deg
        self.altitude = altitude*u.km
        self.isl_dist = isl_dist
        self.dtype = dtype
        
        self.num_planes = num_planes
        self.num_sats_per_plane = num_sats_per_plane
        self.n = num_planes*num_sats_per_plane

        self.sats = []
        self._init_sats()

        #If T > period, we can skip planes if the first sat can't see a task. 
        #If T < period, but can't currently skip planes (but can implement something more clever later if needed)
        self.skip_planes = False 

        #If timestep is 63.79 seconds and altitude is 550 km, then it's 4deg per timestep,
        #and we can reuse the satellite proximities in the plane because they're just offset by 360/num_sats_per_plane/4 time steps.
        self.timestep_offset = None

        self.use_graphs = use_graphs
        self.sat_rs_over_time = None
        self.graphs = None
        if T is not None:
            self.propagate_orbits_and_generate_graphs(T, dt)

    def _init_sats(self):
        a = Earth.R.to(u.km) + self.altitude
        ecc = 0*u.one
        argp = 0*u.deg

        for plane_num in range(self.num_planes):
            raan = plane_num*360/self.num_planes*u.deg
            for sat_num in range(self.num_sats_per_plane):
                ta = (sat_num*360/self.num_sats_per_plane - 180)*u.deg
                sat = Satellite(Orbit.from_classical(Earth, a, ecc, self.inc, raan, argp, ta), plane_id=plane_num)
                self.sats.append(sat)
    
    def reset_orbits(self):
        for sat in self.sats:
            sat.orbit = deepcopy(sat.init_orbit)
    
    def propagate_orbits_and_generate_graphs(self,T, dt=63.76469*u.second):
        """
        Propagate the orbits of all satellites forward in time by T timesteps,
        storing satellite orbits positions over time in a (n x 3 x T) array.

        Proximities are generated later.
        """
        self.reset_orbits() #put satellites back in their initial position if they've moved.

        self.sat_rs_over_time = np.zeros((self.n, 3, T), dtype=self.dtype)
        self.graphs = []
        for k in range(T):
            if self.use_graphs:
                self.graphs.append(self.determine_connectivity_graph())
            else:
                self.graphs.append(None)
            
            for i, sat in enumerate(self.sats):
                sat.propagate_orbit(dt)
                self.sat_rs_over_time[i, :, k] = sat.orbit.r.to_value(u.km)

        if T > sat.orbit.period.to_value(u.min)/dt.to_value(u.min):
            self.skip_planes = True

        if dt == 63.76469*u.second and self.altitude == 550*u.km and (360/self.num_sats_per_plane)%4 == 0:
            self.timestep_offset = -int(360/self.num_sats_per_plane/4)

        return self.graphs
        
    def get_proximities_for_random_tasks(self, m, max_lat=55, fov=60, seed=None):
        """
        Assume tasks are all worth the same amount (1).
        """
        np.random.seed(seed)
        T = self.sat_rs_over_time.shape[2]
        sat_prox_mat = np.zeros((self.n, m, T), dtype=self.dtype)

        #precalculate the sigma which determines how prox falls off with angle
        prox_at_max_fov = 0.05
        gaussian_sigma_2 = np.sqrt(-(fov**2)/(2*np.log(prox_at_max_fov)))**2

        #~~~~~~~~~Generate m random tasks on the surface of earth~~~~~~~~~~~~~
        for j in range(m):
            lon = np.random.uniform(-180, 180)
            lat = np.random.uniform(-max_lat, max_lat)
            task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, Earth).cartesian_cords.to_value(u.km)

            # if self.skip_planes:
            for plane in range(self.num_planes):
                i = plane*self.num_sats_per_plane
                for k in range(T):
                    sat_r = self.sat_rs_over_time[i, :, k]
                    sat_prox_mat[i, j, k] = calc_fov_based_proximities_fast(sat_r, task_loc, fov, gaussian_sigma_2)
                
                if np.max(sat_prox_mat[i, j, :]) == 0 and self.skip_planes:
                    continue

                for other_sat_i in range(1, self.num_sats_per_plane):
                    if self.timestep_offset is not None:
                        sat_prox_mat[i+other_sat_i, j, :] = np.roll(sat_prox_mat[i, j, :], self.timestep_offset*other_sat_i)
                    else:
                        for k in range(T):
                            sat_r = self.sat_rs_over_time[i+other_sat_i, :, k]
                            sat_prox_mat[i+other_sat_i, j, k] = calc_fov_based_proximities_fast(sat_r, task_loc, fov, gaussian_sigma_2)
        
        return sat_prox_mat

    def determine_connectivity_graph(self):
        #Build adjacency matrix
        adj = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1, self.n):
                sat1_r = self.sats[i].orbit.r.to_value(u.km)
                sat2_r = self.sats[j].orbit.r.to_value(u.km)
                R = self.sats[i].orbit._state.attractor.R.to_value(u.km)

                if line_of_sight(sat1_r, sat2_r, R) >=0 and np.linalg.norm(sat1_r-sat2_r) < self.isl_dist:
                    adj[i,j] = 1
                    adj[j,i] = 1

        return nx.from_numpy_array(adj)
    
def calc_fov_based_proximities_fast(sat_r, task_r, fov, gaussian_sigma_2):
    def can_see(sat_r, task_r):
        task_R_norm_2 = np.inner(task_r, task_r)
        proj_sat = np.dot(task_r, sat_r)
        return proj_sat > task_R_norm_2
    
    if can_see(sat_r, task_r):
        sat_to_task = task_r - sat_r

        angle_btwn = np.arccos(np.dot(-sat_r, sat_to_task)/(np.linalg.norm(sat_r)*np.linalg.norm(sat_to_task)))
        angle_btwn *= 57.2957795131 #convert to degrees (180/pi precalculated)

        if angle_btwn < fov:
            task_proximity = np.exp(-(angle_btwn*angle_btwn)/(2*(gaussian_sigma_2)))
        else:
            task_proximity = 0
    else:
        task_proximity = 0
        
    return task_proximity

class Satellite(object):
    def __init__(self, orbit, id=None, plane_id=None, fov=60):
        self.orbit = orbit
        #Can disable if worried about performance, just used for plotting
        self.init_orbit = deepcopy(orbit)

        self.id = id
        self.plane_id = plane_id

        self.fov = fov

    def propagate_orbit(self, time):
        """
        Given a time interval (a astropy quantity object),
        propagates the orbit of the satellite.
        """
        self.orbit = self.orbit.propagate(time)
        return self.orbit