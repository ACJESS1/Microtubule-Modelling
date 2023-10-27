#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:12:05 2023

@author: ale
"""

import itertools
import hoomd
import math

import numpy
import gsd.hoomd
import sphere_fres as sf

N_particles = 25

L = 7
position=[]
for i in range(int(N_particles/5)):
   position.append([[i-2,-2,0],[i-2,-1,0],[i-2,0,0],[i-2,1,0],[i-2,2,0]])

angs=math.sin(math.pi*(1/4))
angc=math.cos(math.pi*(1/4))

orientation = [(1, 0, 0, 0)] * N_particles
#orientation = [(angc, 0, 0, angs)] * N_particles
#orientation = [(angc,angs, 0,  0)] * N_particles
#orientation = [(angc, 0, 0, angs),(1,0,0,0)] 
#orientation = [(0, 0, 1, 0)] * N_particles


# gsd snapshot
snapshot = gsd.hoomd.Frame()
snapshot.particles.N = N_particles
snapshot.particles.position = position
snapshot.particles.orientation = orientation
snapshot.particles.typeid = [0] * N_particles
snapshot.particles.types = ['A']
snapshot.configuration.box = [L, L, L, 0, 0, 0]
with gsd.hoomd.open(name='initial.gsd', mode='w') as f:
    f.append(snapshot)

# build simulation
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=0)
sim.create_state_from_gsd(filename='initial.gsd')

kf_delta_deg = 10  # half-opening angle of patches
kf_epsilon = 1.0  # strength of patchy interaction in kT
kf_lambda = 2.0  # range of patchy interaction
sigma = 4.0  # hard core diameter

q_angle = [math.cos(math.pi*(1/4)), 0, 0, math.sin(math.pi*(1/4))]

#E_long site parameters
a_long = 4.5
b_long = 10
r_long = 13

a_lat = 6.3
b_lat = 8.0
r_lat = 8


mc = hoomd.hpmc.integrate.Sphere(default_d=0.05, default_a=0.1)
mc.shape['A'] = dict(diameter=sigma, orientable=True)
sim.operations.integrator = mc

patch_code = f"""
const float delta = {kf_delta_deg} * M_PI / 180;  // delta in radians
const float epsilon = {kf_epsilon:f};
const float lambda = {kf_lambda:f};
const float sigma = {sigma:f};  // hard core diameter

const float a_o = {a_long};
const float b_o = {b_long};
const float r_o = {r_long};

const float a_a = {a_lat};
const float b_a = {b_lat};
const float r_a = {r_lat};


const float kT = param_array[0];

const const vec3<float> temp(0, 0, sqrtf(2)/2);
const quat<float> q_perp(sqrtf(2)/2,temp);

const vec3<float> ehat_particle_reference_frame(1, 0, 0);
vec3<float> ehat_i = rotate(q_i, ehat_particle_reference_frame);
vec3<float> ehat_j = rotate(q_j, ehat_particle_reference_frame);

vec3<float> ehat_i_p = rotate(q_perp*q_i, ehat_particle_reference_frame);
vec3<float> ehat_j_p = rotate(q_perp*q_j, ehat_particle_reference_frame);

vec3<float> r_hat_ij = r_ij / sqrtf(dot(r_ij, r_ij));
bool patch_on_i_is_aligned_with_r_ij = dot(ehat_i, r_hat_ij) >= cos(delta);
bool patch_on_j_is_aligned_with_r_ji = dot(ehat_j, -r_hat_ij) >= cos(delta);
bool patch_on_i_is_aligned_with_r_ji = dot(ehat_i, -r_hat_ij) >= cos(delta);
bool patch_on_j_is_aligned_with_r_ij = dot(ehat_j, r_hat_ij) >= cos(delta);

bool patch_on_i_is_perpendicular_r_ij = abs(dot(ehat_i_p, r_hat_ij)) >= cos(delta);
bool patch_on_j_is_perpendiculat_r_ij = abs(dot(ehat_j_p, r_hat_ij)) >= cos(delta);

float rsq = dot(r_ij, r_ij);
float r_ij_length = sqrtf(rsq);
float r_ij_ls = r_ij_length - sigma;
if ((patch_on_i_is_aligned_with_r_ij || patch_on_i_is_aligned_with_r_ji)
    && (patch_on_j_is_aligned_with_r_ji || patch_on_j_is_aligned_with_r_ij)
    && r_ij_length < lambda*sigma)
    {{
    return ((a_o* pow(r_ij_ls/r_o,2)*exp(-r_ij_ls/r_o))
        -b_o*exp(-(pow(r_ij_ls/r_o,2))) ) / kT;
    }}
else if (patch_on_i_is_perpendicular_r_ij
         && patch_on_j_is_perpendiculat_r_ij
         && r_ij_length < lambda*sigma)
    {{
    return ((a_a* pow(r_ij_ls/r_a,2)*exp(-r_ij_ls/r_a))
    -b_a*exp(-(pow(r_ij_ls/r_a,2))) ) / kT;      
    }}
else
    {{
    return 0.0;
    }}
"""

#(a*(x/r)**2*math.exp(-x/r)-b*math.exp(-(x/r)**2));

r_cut = sigma + sigma * (kf_lambda - 1)
initial_kT = 0.0
patch_param_array = [initial_kT]

patch_potential = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                                    code=patch_code,
                                                    param_array=[initial_kT])

mc.pair_potential = patch_potential
sim.run(30000)

with sim.state.cpu_local_snapshot as sn:
    #print(sn.particles.typeid)
    #print(sn.particles.position)
    #print(sn.particles.orientation)
    dist =sn.particles.position[0,:]- sn.particles.position[1,:]
    #print(math.sqrt(numpy.dot(dist, dist)))
print(patch_potential.energy)