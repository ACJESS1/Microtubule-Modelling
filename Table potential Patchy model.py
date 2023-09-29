#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:22:26 2023

@author: ale
"""

import matplotlib.pyplot as plt
import itertools
import hoomd
import math

import xarray as xr
import numpy
import gsd.hoomd
import sphere_fres as sf

N_particles =25

L = 100
position=[]
for i in range(int(N_particles/5)):
   position.append([[6*(i-2),6*(-2),0],[6*(i-2),6*(-1),0],[6*(i-2),0,0],[6*(i-2),6*(1),0],[6*(i-2),6*(2),0]])

angs=math.sin(math.pi*(1/4))
angc=math.cos(math.pi*(1/4))

#N_particles = 2
#position = [[6,0,0],[0,0,0]]

orientation = [(1, 0, 0, 0)] * N_particles



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

#simulation parameters
kf_lambda = 2.0  # range of patchy interaction
sigma = 1.0  # hard core diameter
initial_kT = 0.0001

#The terrible things done to the data set

ds_disk = xr.open_dataset("mt_full.nc")
temp1 = ds_disk.to_array()
tarr = temp1.data
tarr = numpy.squeeze(tarr)
tarrt = numpy.reshape(tarr,(11*11*11*2*2*2))
patch_param_array = numpy.append(tarrt,initial_kT)



mc = hoomd.hpmc.integrate.Sphere(default_d=2, default_a=0.2)
mc.shape['A'] = dict(diameter=sigma, orientable=True)
sim.operations.integrator = mc


patch_code = f"""

quat<float> q_rel = q_j*conj(q_i);
vec3<float> angles(0,0,0);

// roll (x-axis rotation)
    double sinr_cosp = 2 * (q_rel.s * q_rel.v.x + q_rel.v.y * q_rel.v.z);
    double cosr_cosp = 1 - 2 * (q_rel.v.x * q_rel.v.x + q_rel.v.y * q_rel.v.y);
    angles.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = std::sqrt(1 + 2 * (q_rel.s * q_rel.v.y - q_rel.v.x * q_rel.v.z));
    double cosp = std::sqrt(1 - 2 * (q_rel.s * q_rel.v.y - q_rel.v.x * q_rel.v.z));
    angles.y = 2 * std::atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q_rel.s * q_rel.v.z + q_rel.v.x * q_rel.v.y);
    double cosy_cosp = 1 - 2 * (q_rel.v.y * q_rel.v.y + q_rel.v.z * q_rel.v.z);
    angles.z = std::atan2(siny_cosp, cosy_cosp);

const int sx=11,sy=11,sz=11,sa=2,sb=2,sc=2;

int x = round(r_ij.x/2+5);
int y = round(r_ij.y/2+5);
int z = round(r_ij.z/2+5);

//int a=0,b=0,c=0;

int a=round((-abs(angles.x)/M_PI)+1);
int b=round((-(2*abs(angles.z))/M_PI)+1) ;
int c=round((-abs(angles.z)/M_PI)+1);

const float kT = param_array[sx*sy*sz*sa*sb*sc];

return param_array[x*sy*sz*sa*sb*sc+y*sz*sa*sb*sc+z*sa*sb*sc+a*sb*sc+b*sc+c]/kT;

"""

#(a*(x/r)**2*math.exp(-x/r)-b*math.exp(-(x/r)**2));

r_cut = sigma + sigma * (kf_lambda - 1)

#patch_param_array = [initial_kT,1,2,3]

patch_potential = hoomd.hpmc.pair.user.CPPPotential(r_cut=10,
                                                    code=patch_code,
                                                    param_array=patch_param_array)

mc.pair_potential = patch_potential
sim.run(0000)
print(patch_potential.energy)

#with sim.state.cpu_local_snapshot as sn:
    #print(sn.particles.typeid)
    #print(sn.particles.position)
    #print(sn.particles.orientation)
    #dist =sn.particles.position[0,:]- sn.particles.position[1,:]
    #print(math.sqrt(numpy.dot(dist, dist)))

logger = hoomd.logging.Logger()
logger.add(patch_potential, quantities=[
    'energy',
])
logger[('kT',)] = (lambda: patch_potential.param_array[10648], 'scalar')
gsd_writer = hoomd.write.GSD(filename='log.gsd',
                             trigger=hoomd.trigger.Periodic(50),
                             mode='wb',
                             filter=hoomd.filter.Null(),
                             logger=logger)
sim.operations.writers.append(gsd_writer)

sim.run(5000)

gsd_writer.flush()

data = gsd.hoomd.read_log('log.gsd')
step = data['configuration/step']

beta_potential_energy = data['log/hpmc/pair/user/CPPPotential/energy']
kT = data['log/kT']
potential_energy = beta_potential_energy * kT

plt.close('all')

plt.figure(1)
plt.plot(step[potential_energy<0],potential_energy[potential_energy<0])
#plt.plot(step,potential_energy)
#ax2=ax.twinx()
#ax2.plot(step,kT)

with sim.state.cpu_local_snapshot as sn:
    print(sn.particles.position)
    distlog = numpy.array([])
    for i in range(N_particles):
        for j in range(i+1,N_particles):
            displacement = sn.particles.position[i,:]- sn.particles.position[j,:]
            distlog = numpy.append(distlog,math.sqrt(numpy.dot(displacement, displacement)))
    dist =sn.particles.position[0,:]- sn.particles.position[1,:]
    #print(math.sqrt(numpy.dot(dist, dist)))
    
plt.figure(2)
plt.hist(distlog,bins=int(numpy.max(distlog)))