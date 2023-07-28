#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:59:59 2023

@author: hyeonghun
"""

__author__ = 'Hyeonghun Kim'
__email__ = 'hyk049@ucsd.edu'

import matplotlib.pyplot as plt
import numpy as np
from math import pi

def sph2carte(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)

    y = r * np.cos(phi) * np.sin(theta)

    z = r * np.cos(phi)
        
    return x, y, z



print("The cartesian coordinate of (r, theta, phi) = (1, 0, 0) is ",
      sph2carte(1, 0, 0))
print("The cartesian coordinate of (r, theta, phi) = (1, pi, pi) is ",
      sph2carte(1, pi, pi))
print("The cartesian coordinate of (r, theta, phi) = (1, 2pi, 2pi) is ",
      sph2carte(1, 2*pi, 2*pi))
print("The cartesian coordinate of (r, theta, phi) = (1, -pi, -2pi) is ",
      sph2carte(1, -pi, -2*pi))
print("The cartesian coordinate of (r, theta, phi) = (1, -2pi, -pi) is ",
      sph2carte(1, -2*pi, -pi))
print("The cartesian coordinate of (r, theta, phi) = (1, 2pi, 0.5*pi) is ",
      sph2carte(1, 2*pi, 0.5*pi))

# =============================================================================
# %matplotlib qt
# =============================================================================

fig = plt.figure()
axes  = fig.add_subplot(projection='3d')
r     = np.linspace(0, 1)
theta = np.linspace(0, 2*pi)
phi = np.linspace(0, 2*pi)
x, y, z = sph2carte(r, theta, phi)
axes.plot(x, y, z)


assert np.allclose(sph2carte(1, 0, 0), (0,0,1), atol = 1e-10)

assert np.allclose(sph2carte(1, pi, pi), (0,0,-1), atol = 1e-10)

assert np.allclose(sph2carte(1, 2*pi, 2*pi), (0,0,1), atol = 1e-10)

assert np.allclose(sph2carte(1, -pi, -2*pi), (0,0,1), atol = 1e-10)

assert np.allclose(sph2carte(1, -2*pi, -pi), (0,0,-1), atol = 1e-10)

assert np.allclose(sph2carte(1, 2*pi, 0.5*pi), (1,0,0), atol = 1e-10)

