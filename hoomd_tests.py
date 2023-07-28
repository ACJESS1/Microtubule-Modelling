#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:33:06 2023

@author: ale
"""
import itertools
import hoomd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gsd.hoomd
import octahedron_fresnel as octf
import oct_frame_fresnel as octff

cpu = hoomd.device.CPU()

sim = hoomd.Simulation(device=cpu)

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape['octahedron'] = dict(vertices=[
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
])
mc.nselect = 2
mc.d['octahedron'] = 0.15
mc.a['octahedron'] = 0.2

sim.operations.integrator = mc

m = 3
N_particles = 2 * m**3
spacing = 1.2
K = math.ceil(N_particles**(1 / 3))
L = K * spacing

x = np.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
position = position[0:N_particles]

orientation = [(1, 0, 0, 0)] * N_particles

print(L)


frame = gsd.hoomd.Frame()
frame.particles.N = N_particles
frame.particles.position = position
frame.particles.orientation = orientation
frame.particles.typeid = [0] * N_particles
frame.particles.types = ['octahedron']
frame.configuration.box = [L, L, L, 0, 0, 0]

with gsd.hoomd.open(name='lattice.gsd', mode='w') as f:
    f.append(frame)

sim.create_state_from_gsd(filename='lattice.gsd')
