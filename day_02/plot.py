#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:51:08 2023

@author: hyeonghun
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1)
plt.plot(x, np.exp(x))
plt.xlabel(r'$0 \leq x<1$')
plt.ylabel('$e^x$')
plt.title('Exponential function')
plt.show()