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
import h3
from shapely.geometry import Polygon

class StarlinkSim(object):
    def __init__(self, T=None, dt=63.76469*u.second, dtype=np.float32) -> None:
        self.dtype = dtype

        self.sats = []
        self._init_starlink_sats()

        # #If T > period, we can skip planes if the first sat can't see a task. 
        # #If T < period, but can't currently skip planes (but can implement something more clever later if needed)
        # self.skip_planes = False 

        # #If timestep is 63.79 seconds and altitude is 550 km, then it's 4deg per timestep,
        # #and we can reuse the satellite proximities in the plane because they're just offset by 360/num_sats_per_plane/4 time steps.
        # self.timestep_offset = None

        self.sat_rs_over_time = None
        if T is not None:
            self.propagate_orbits(T, dt)

    def _init_starlink_sats(self):
        ecc = 0*u.one
        argp = 0*u.deg

        self.all_num_planes = [32, 32, 8, 5, 6]
        self.all_num_sats_per_planes = [50, 50, 50, 75, 75]
        all_incs = [53, 53.8, 74, 81, 70]
        all_alts = [550, 510, 530, 675, 725]
        plane_group = 0
        for (num_planes, num_sats_per_plane, inc, alt) in zip(self.all_num_planes, self.all_num_sats_per_planes, all_incs, all_alts):
            num_planes = num_planes
            num_sats_per_plane = num_sats_per_plane
            inc = inc*u.deg
            a = Earth.R.to(u.km) + alt*u.km

            for plane_num in range(num_planes):
                raan = (plane_num/num_planes)*360*u.deg
                for sat_num in range(num_sats_per_plane):
                    ta = ((sat_num/num_sats_per_plane)*360 - 180)*u.deg
                    sat = Satellite(Orbit.from_classical(Earth, a, ecc, inc, raan, argp, ta), plane_id=(plane_group, plane_num))
                    self.sats.append(sat)

            plane_group += 1

        self.n = len(self.sats)

        # a = Earth.R.to(u.km) + 550*u.km
        # ecc = 0*u.one
        # argp = 0*u.deg

        # for plane_num in range(18):
        #     raan = plane_num*360/18*u.deg
        #     for sat_num in range(18):
        #         ta = (sat_num*360/18 - 180)*u.deg
        #         sat = Satellite(Orbit.from_classical(Earth, a, ecc, 58*u.deg, raan, argp, ta), plane_id=plane_num)
        #         self.sats.append(sat)

        # self.n = len(self.sats)
    
    def reset_orbits(self):
        for sat in self.sats:
            sat.orbit = deepcopy(sat.init_orbit)
    
    def propagate_orbits(self,T, dt=63.76469*u.second, test=False):
        """
        Propagate the orbits of all satellites forward in time by T timesteps,
        storing satellite orbits positions over time in a (n x 3 x T) array.

        Proximities are generated later.
        """
        self.reset_orbits() #put satellites back in their initial position if they've moved.

        self.sat_rs_over_time = np.zeros((self.n, 3, T), dtype=self.dtype)
        for k in range(T):
            print(f"Propagating orbits, time {k+1}/{T}...",end='\r')
            for i, sat in enumerate(self.sats):
                sat.propagate_orbit(dt)
                self.sat_rs_over_time[i, :, k] = sat.orbit.r.to_value(u.km)
        print("")
        #technically this will only check the most recent sat, not a sat for all planes
        self.skip_planes = False
        if T > sat.orbit.period.to_value(u.min)/dt.to_value(u.min):
            self.skip_planes = True
    
    def get_proximities_for_coverage_tasks(self, res=2, max_lat=70, fov=60, seed=None):
        """
        Assume tasks are all worth the same amount (1).

        res is a metric from the H3 package which defines the resolution of the hexagons,
        with resolution 1 being the highest resolution.
            res 1: 721 at 55 max_lat, 824 at 70 max_lat
            res 2: 4908 at 55 max_lat, 5000+ at 70 max_lat, so prob the right amount for Starlink w/ 4200 sats
        """
        np.random.seed(seed)
        T = self.sat_rs_over_time.shape[2]

        #precalculate the sigma which determines how prox falls off with angle
        prox_at_max_fov = 0.05
        gaussian_sigma_2 = np.sqrt(-(fov**2)/(2*np.log(prox_at_max_fov)))**2

        hexagons = generate_smooth_coverage_hexagons((-max_lat, max_lat), (-180, 180), res=res)
        hex_to_task_mapping = {hexagon: i for i, hexagon in enumerate(hexagons)}
        m = max(self.n, len(hexagons)) #if there are more sats than hexagons, we need at least n tasks, so we make dummy ones

        sat_prox_mat = np.zeros((self.n, m, T), dtype=self.dtype)
        #Add tasks at centroid of all hexagons
        for j, hexagon in enumerate(hexagons):
            print("Calculated proximity of tasks to satellites, task {}/{}...".format(j+1, len(hexagons)),end='\r')
            boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
            polygon = Polygon(boundary)

            lat = polygon.centroid.y
            lon = polygon.centroid.x

            task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, Earth).cartesian_cords.to_value(u.km)

            tot_i = 0
            for num_planes, num_sats_per_plane in zip(self.all_num_planes, self.all_num_sats_per_planes):
                for plane in range(num_planes):
                    i = tot_i + plane*num_sats_per_plane
                    for k in range(T):
                        sat_r = self.sat_rs_over_time[i, :, k]
                        sat_prox_mat[i, j, k] = calc_fov_based_proximities_fast(sat_r, task_loc, fov, gaussian_sigma_2)
                    if np.max(sat_prox_mat[i, j, :]) == 0 and self.skip_planes:
                        continue

                    for other_sat_i in range(1, num_sats_per_plane):
                        for k in range(T):
                            sat_r = self.sat_rs_over_time[i+other_sat_i, :, k]
                            sat_prox_mat[i+other_sat_i, j, k] = calc_fov_based_proximities_fast(sat_r, task_loc, fov, gaussian_sigma_2)

                tot_i += num_planes*num_sats_per_plane

        return sat_prox_mat
    
    def get_proximities_for_random_tasks(self, m, max_lat=70, fov=60, seed=None):
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

            tot_i = 0
            for num_planes, num_sats_per_plane in zip(self.all_num_planes, self.all_num_sats_per_planes):
                for plane in range(num_planes):
                    i = tot_i + plane*num_sats_per_plane
                    for k in range(T):
                        sat_r = self.sat_rs_over_time[i, :, k]
                        sat_prox_mat[i, j, k] = calc_fov_based_proximities_fast(sat_r, task_loc, fov, gaussian_sigma_2)
                    if np.max(sat_prox_mat[i, j, :]) == 0 and self.skip_planes:
                        continue

                    for other_sat_i in range(1, num_sats_per_plane):
                        for k in range(T):
                            sat_r = self.sat_rs_over_time[i+other_sat_i, :, k]
                            sat_prox_mat[i+other_sat_i, j, k] = calc_fov_based_proximities_fast(sat_r, task_loc, fov, gaussian_sigma_2)

                tot_i += num_planes*num_sats_per_plane
                
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
    
def generate_smooth_coverage_hexagons(lat_range, lon_range, res=1):
    # Initialize an empty set to store unique H3 indexes
    hexagons = set()

    # Step through the defined ranges and discretize the globe
    lat_steps, lon_steps = 0.2/res, 0.2/res
    lat = lat_range[0]
    while lat <= lat_range[1]:
        lon = lon_range[0]
        while lon <= lon_range[1]:
            # Find the hexagon containing this lat/lon
            hexagon = h3.geo_to_h3(lat, lon, res)
            hexagons.add(hexagon)
            lon += lon_steps
        lat += lat_steps
        
    return list(hexagons) #turn into a list so that you can easily index it later