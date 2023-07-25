#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:19:27 2023

@author: ale
"""

import itertools
import math

import gsd.hoomd
import hoomd
import numpy
import sphere_fres as sf #slightly modified rendering code taken from Hoomd tutorials

frame = gsd.hoomd.Frame()

# Place proteins in the box.
N_cores = 2#number of proteins, must be even to create whole number of dimers
frame.particles.N = N_cores*5
#set site positions relative to central sphere
p_pos=[]
for i in range(N_cores):
    p_pos.append([[i,0,0],[i-0.5,0,0],[i,0.5,0],[i,-0.5,0],[i+0.5,0,0]])

frame.particles.position = p_pos
frame.particles.orientation = [(1,0,0,0)]*N_cores*5
frame.particles.types = ['core','Lo','La','B']
#types represent Protein, Longitude site, Latitude site, and dimer Bond site
frame.particles.typeid = [0,1,2,2,3]*N_cores
frame.configuration.box = [7,7,7, 0, 0, 0]

#Creates all bonds for a Dimer
frame.bonds.N=int((N_cores/2)*9)
frame.bonds.types = ['dimer','fix']
b_typeid=[]
b_group=[]
n=0
for i in range(int(N_cores/2)):
    n=int(10*i)
    b_typeid.append([1,1,1,1,1,1,1,1,0])
    b_group.append([[n,n+1],[n,n+2],[n,n+3],[n,n+4],
                    [n+5,n+1+5],[n,n+2+5],[n,n+3+5],[n,n+4+5],
                    [n+4,n+9]])
frame.bonds.typeid = b_typeid
frame.bonds.group = b_group

#Creates all angle constaints
frame.angles.N = int((N_cores/2)*9)
frame.angles.types= ['fixs','fixr','dimer']
a_typeid=[]
a_group=[]
n=0
for i in range(int(N_cores/2)):
    n=int(10*i)
    a_typeid.append([0,0,1,1,2,0,0,1,1])
    a_group.append([[n+1,n,n+4],[n+2,n,n+3],[n+1,n,n+2],[n+3,n,n+4],
                    [n,n+4,n+5],
                    [n+1+5,n+5,n+4+5],[n+2+5,n+5,n+3+5],[n+1+5,n+5,n+2+5],[n+3+5,n+5,n+4+5]])
frame.angles.typeid = a_typeid
frame.angles.group = a_group

#Improper forces all sites in a dimer to be planar
frame.impropers.N = int(N_cores/2)
frame.impropers.types = ['Planar']
i_typeid=[]
i_group=[]
n=0
for i in range(int(N_cores/2)):
    n=int(i)
    i_typeid.append([0])
    i_group.append([n+2,n,n+5,n+5+3])
frame.impropers.typeid = i_typeid
frame.impropers.group = i_group



with gsd.hoomd.open(name='particles.gsd', mode='w') as f:
    f.append(frame)

#defines all bons, angle and improper forces
# 'fix' indicatres bonds and angles meant to constain positions, so higher k value
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['dimer'] = dict(k=200, r0=0.0)
harmonic.params['fix'] = dict(k=2000, r0=0.5)

harmonic_a = hoomd.md.angle.Harmonic()
harmonic_a.params['dimer'] = dict(k=2000, t0=math.pi)
harmonic_a.params['fixs'] = dict(k=2000, t0=math.pi)
harmonic_a.params['fixr'] = dict(k=2000, t0=math.pi/2)

harmonic_i = hoomd.md.improper.Harmonic()
harmonic_i.params['Planar'] = dict(k=2000, chi0 = math.pi)

#functions for Potential and force of longitude and latitude sites
Get_pot = lambda a,b,r,x : a*(x/r)**2*math.exp(-x/r)-b*math.exp(-(x/r)**2)
Get_force = lambda a,b,r,x : (x*math.exp(-(x*(r + x))/r**2)*(-a*math.exp(x**2/r**2)*(2*r - x) - 2*b*r*math.exp(x/r)))/r**3

#Sample values of potential and force to use in pair.Table object
#Note upper limit of R should be the same as the r_cut value for accurate recreation
R= numpy.linspace(0,1,100)
U_list_o = []
F_list_o = []
U_list_a = []
F_list_a = []
for i in R:
    U_list_o.append(Get_pot(1,1,0.05,i))
    F_list_o.append(Get_force(1,1,0.05,i))
    U_list_a.append(Get_pot(1,1,0.05,i))
    F_list_a.append(Get_force(1,1,0.05,i))
    

cell = hoomd.md.nlist.Cell(buffer=0)  

#Site pair potentials defined so that sites interact only with the same types
#i.e. r_cut and potentials set to 0
E_site = hoomd.md.pair.Table(nlist = cell,default_r_cut=0)
E_site.params[('Lo','Lo')]= dict(r_min=0, U=U_list_o, F=F_list_o)
E_site.r_cut[('Lo','Lo')]= 1

E_site.params[('La','La')]= dict(r_min=0, U=U_list_a, F=F_list_a)
E_site.r_cut[('La','La')]= 1

E_site.params[(['B','core'],['core','Lo','La','B'])]= dict(r_min=0, U=[0], F=[0])
E_site.r_cut[(['B','core'],['core','Lo','La','B'])]= 0

E_site.params[('Lo','La')]= dict(r_min=0, U=[0], F=[0])
E_site.r_cut[('Lo','La')]= 0

#lennard jones interaction between core spheres with sigma=r_cut
#This creates a hardshell to prevent overlap without adding further attractive interactions
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('core', 'core')] = dict(epsilon=10, sigma=1)
lj.r_cut[('core', 'core')] = 1

lj.params[(['Lo','La','B'],['core','Lo','La','B'])] = dict(epsilon=0, sigma=0)
lj.r_cut[(['Lo','La','B'],['core','Lo','La','B'])] = 0




#create sim and add forces and methods
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
sim.create_state_from_gsd(filename='particles.gsd')
brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.01)
#lang =hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=15)
integrator = hoomd.md.Integrator(dt=0.000005,
                                 methods=[brownian],
                                 forces=[harmonic,harmonic_a,harmonic_i,E_site,lj])
sim.operations.integrator = integrator

#Runs sim and outputs positions and typeID for diagnostic
with sim.state.cpu_local_snapshot as sn:
    print(sn.particles.typeid)
    print(sn.particles.position)
sim.run(0)

sim.run(10)

with sim.state.cpu_local_snapshot as sn:
    print(sn.particles.typeid)
    print(sn.particles.position)