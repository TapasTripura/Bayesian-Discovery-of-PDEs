#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:40:30 2022

@author: user
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

from fenics import *
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
set_log_active(False)   # To stop the display "solving variational problem"

T = 2.0            # final time
num_steps = 2000     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 63
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

u_D = Constant(0)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
                 degree=2, a=5)
u_n = interpolate(u_0, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Create VTK file for saving solution
# vtkfile = File('heat_gaussian/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
sol = np.zeros([num_steps,nx+1,ny+1]) 
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bc)

    # Save to file and plot solution
    # vtkfile << (u, t)
    # plot(u)
    sol[n,:,:] = u.compute_vertex_values().reshape(nx+1,ny+1)
    
    # Compute error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector() - u.vector()).max()
    print('iteration = %0.2f: t = %.2f: error = %.3g' % (n, t, error))

    # Update previous solution
    u_n.assign(u)

# %%
""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fig1 = plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.3)

index = 0
for i in range(num_steps):
    if i % 125 == 0:
        plt.subplot(4,4,index+1)
        plt.imshow(sol[i,:,:], origin='lower', cmap='jet', 
                   vmin=0, vmax=0.02, extent=[0,1,0,1])
        plt.colorbar(fraction=0.045)
        index = index+1

# sio.savemat('data/heat_centersource_1000Hz_2sec_64x_0to1.mat', mdict={'sol': sol})

# %%
""" Save the animation """

# def plotheatmap(u_k, k):
#     # Clear the current plot figure
#     plt.clf()

#     plt.title(f"Temperature at t = {k*dt:.3f} unit time")
#     plt.xlabel("x")
#     plt.ylabel("y")

#     # This is to plot u_k (u at time-step k)
#     plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=0.02)
#     plt.colorbar()
#     return plt

# def animate(k):
#     plotheatmap(sol[k,:,:], k)

# anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=num_steps, repeat=False)
# anim.save("heat_Gaussian_solution.gif")
