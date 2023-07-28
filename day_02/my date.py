#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:30:55 2023

@author: hyeonghun day_02
"""

import matplotlib.pyplot as plt
import numpy as np


from datetime import datetime

from swmfpy.web import get_omni_data

start_time = datetime(1996, 2, 21)

end_time = datetime(1996, 2, 22)

data = get_omni_data(start_time, end_time)


data.keys()


t = data['times']


#%matplotlib qt
plt.plot(t, data['al'])
plt.xlabel('time')
plt.ylabel('AL')
plt.title('AL in my birthday of 1996 Feb. 21st')
plt.show()

