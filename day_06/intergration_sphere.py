#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:17:17 2023

@author: hyeonghun
"""

import numpy as np
import matplotlib.pyplot as plt

n_0 = pow(10,19)
alt_0 = 100
alt_n = 500
nPts = 100

r_0 = 6370

T_0 = 200
T_n = 1000

T = np.linspace(T_0, T_n, nPts)
alt = np.linspace(alt_0, alt_n, nPts)

r = r_0

g = 3.99*pow(10,14)/(((r + alt)*1000)**2)

m = 28 * 1.67*pow(10, -27)
k = 1.38 * pow(10,-23)

H = k*T/(m*g) /1000

dz = alt[1] - alt[0]

n_prev = n_0

n = np.zeros(len(alt))
n[0] = n_0

for ii in range(len(alt)-1):
    temp = T[ii]/T[ii+1] * n_prev * np.exp(-dz/H[ii])
    n[ii+1] = temp
    n_prev = temp
    


plt.plot(alt, np.log(n))
plt.xlabel('Altitude [km]')
plt.ylabel('n')

plt.show()