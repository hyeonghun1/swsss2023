#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:18:29 2023

@author: hyeonghun

"""

import numpy as np
import matplotlib.pyplot as plt

num_x = 10

num_y = 20

x = np.linspace(0,1,num_x)

y = np.linspace(0,1,num_y)

z = np.random.randn(num_y, num_x)

plt.pcolormesh(x,y,z)
plt.colorbar()