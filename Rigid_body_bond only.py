#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:25:11 2023

@author: ale
"""

import itertools
import math

import gsd.hoomd
import hoomd
import numpy
import sphere_fres as sf

frame = gsd.hoomd.Frame()

# Define core protien properties
N_cores = 2
frame.particles.N = N_cores
frame.particles.position = [[0,0,0],[1,0,0]]
frame.particles.orientation = [(1,0,0,0)]*N_cores
frame.particles.types = ['core','Lo','La']
#define types for interactions sites for later use
frame.particles.typeid = [0]*N_cores
frame.particles.mass=[1]*N_cores
frame.configuration.box = [11, 11, 11, 0, 0, 0]
frame.particles.moment_inertia = [0.4,0.4,0.4]*N_cores

#frame.bonds.N = 1
#frame.bonds.types = ['H-H']
#frame.bonds.typeid = [0]
#frame.bonds.group = [0, 1]

#frame.angles.N = 0
#frame.angles.types = ['c-Lo-c']
#frame.angles.typeid = []
#frame.angles.group = []




rigid = hoomd.md.constrain.Rigid()
#rigid body constraint for particles
site_pos = [(0.5,0,0),(0,0.5,0),(0,-0.5,0)]
rigid.body['core'] = {
    "constituent_types": ['Lo','La','La'],
    "positions": site_pos,
    "orientations": [(1.0, 0.0, 0.0, 0.0)]*3
}

with gsd.hoomd.open(name='particle_centers.gsd', mode='w') as f:
    f.append(frame)

#Create simulation
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=4)
sim.create_state_from_gsd(filename='particle_centers.gsd')
#rigid bodies created in simulation
rigid.create_bodies(sim.state)

integrator = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=True)
integrator.rigid = rigid
sim.operations.integrator = integrator

kT = 0.1
rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
nvt = hoomd.md.methods.ConstantVolume(
    filter=rigid_centers_and_free,
    thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT))
integrator.methods.append(nvt)

sim.run(0)
#%%

#Site potentials definition

Get_pot = lambda a,b,r,x : a*(x/r)**2*math.exp(-x/r)-b*math.exp(-(x/r)**2)
Get_force = lambda a,b,r,x : (x*math.exp(-(x*(r + x))/r**2)*(-a*math.exp(x**2/r**2)*(2*r - x) - 2*b*r*math.exp(x/r)))/r**3

R= numpy.linspace(0,5,100)
U_list_o = []
F_list_o = []
U_list_a = []
F_list_a = []
for i in R:
    U_list_o.append(Get_pot(0,4,2,i))
    F_list_o.append(Get_force(0,4,2,i))
    U_list_a.append(Get_pot(0,4,2,i))
    F_list_a.append(Get_force(0,4,2,i))
    

cell = hoomd.md.nlist.Cell(buffer=0, exclusions=['body'])  
#Site pair potentials defined so that sites interact only with the same types
#i.e. r_cut and potentials set to 0

E_site = hoomd.md.pair.Table(nlist = cell,default_r_cut=0)
E_site.params[('Lo','Lo')]= dict(r_min=0, U=U_list_o, F=F_list_o)
E_site.r_cut[('Lo','Lo')]= 5

E_site.params[('La','La')]= dict(r_min=0, U=U_list_a, F=F_list_a)
E_site.r_cut[('La','La')]= 3

E_site.params[('core',['core','Lo','La'])]= dict(r_min=0, U=[0], F=[0])
E_site.r_cut[('core',['core','Lo','La'])]= 0

E_site.params[('Lo','La')]= dict(r_min=0, U=[0], F=[0])
E_site.r_cut[('Lo','La')]= 0

#lennard jones interaction between core spheres with sigma=r_cut
#This creates a hardshell to prevent overlap without adding further attractive interactions

lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('core', 'core')] = dict(epsilon=10, sigma=1)
lj.r_cut[('core', 'core')] = 1

lj.params[('Lo',['core','Lo','La'])] = dict(epsilon=0, sigma=0)
lj.r_cut[('Lo',['core','Lo','La'])] = 0

lj.params[('La',['core','La'])] = dict(epsilon=0, sigma=0)
lj.r_cut[('La',['core','La'])] = 0


integrator.forces.append(E_site)
integrator.forces.append(lj)


sim.state.thermalize_particle_momenta(filter=rigid_centers_and_free, kT=kT)
sim.run(0)


sim.run(1000)

with sim.state.cpu_local_snapshot as sn:
    print(sn.particles.typeid)
    print(sn.particles.position)
    
#attempts to edit angle data

temp = sim.state.get_snapshot()
#if temp.communicator.rank == 0:
#temp.angles.N = 1
#temp.angles.types = ['c-Lo-c']
#temp.angles.typeid[0] = 0
#temp.angles.group[0] = [0, 1, 2]

#sim.state.set_snapshot(temp)

#harmonic_a = hoomd.md.angle.Harmonic()
#harmonic_a.params['c-Lo-c'] = dict(k=400.0, t0=math.pi)

#integrator.forces.append(harmonic_a)

#with gsd.hoomd.open(name='add_angles.gsd', mode='r+') as f:
    #f.angles.N = 1
    #f.angles.types = ['c-Lo-c']
    #f.angles.typeid = [0]
    #f.angles.group = [0,1,3]


#sim.create_state_from_gsd(filename='add_angles.gsd')

#sim.run(1000)
