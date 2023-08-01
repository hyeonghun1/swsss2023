#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:16:28 2023

@author: hyeonghun
"""

from scipy.optimize import fsolve 
import numpy as np
# right -hand -side
def f(x):
    return np.exp(x) - 4*x
# initial guess
x0 = 10
# solve statment assings solution to x
x = fsolve(f, x0)