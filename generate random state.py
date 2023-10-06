#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:40:42 2023

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


N_particles = 200

spacing = 20
K = math.ceil(N_particles**(1 / 3))
L = K * spacing

x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
position = position[0:N_particles]

orientation = [(1, 0, 0, 0)] * N_particles

snapshot = gsd.hoomd.Frame()
snapshot.particles.N = N_particles
snapshot.particles.position = position
snapshot.particles.orientation = orientation
snapshot.particles.typeid = [0] * N_particles
snapshot.particles.types = ['A']
snapshot.configuration.box = [L, L, L, 0, 0, 0]
with gsd.hoomd.open(name='initial_cube.gsd', mode='w') as f:
    f.append(snapshot)
    
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=0)
sim.create_state_from_gsd(filename='initial_cube.gsd')

mc = hoomd.hpmc.integrate.Sphere(default_d=2, default_a=0.2)
mc.shape['A'] = dict(diameter=6, orientable=True)
sim.operations.integrator = mc

sim.run(10000)

hoomd.write.GSD.write(state=sim.state, mode='wb', filename='random.gsd')