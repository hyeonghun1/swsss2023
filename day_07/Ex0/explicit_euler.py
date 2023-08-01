#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:41:10 2023

@author: holtorf
"""

import numpy as np
import matplotlib.pyplot as plt

# Use Euler's method with different stepsizes to solve the IVP:
# dx/dt = -2*x, with x(0) = 3 over the time-horizon [0,2]

# Compare the numerical approximation of the IVP solution to its analytical
# solution by plotting both solutions in the same figure. 


x_0 = 3     # I.C

h = 0.01    # time step

Tfinal = 2

time = np.linspace(0,2,201)

x_current = x_0

x = np.zeros(len(time))

x[0] = x_0

for ii in range(len(time)-1):
    x_next = x_current + h * (-2*x_current)
    
    x[ii+1] = x_next
    
    x_current = x_next
    
    
    
    
#%% Analytical soln
    
x_analytical = 3*np.exp(-2*time)


#%% plot
plt.plot(time, x)
plt.plot(time, x_analytical, color = 'green',linestyle = 'dashed', linewidth = 2.0)
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Time evolution of x(t)')
plt.legend(['numerical', 'analytical'])


