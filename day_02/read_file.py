#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:52:43 2023

@author: hyeonghun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

f = open('/Users/hyeonghun/Desktop/swsss2023/swsss2023/day_02/omni_test.lst')
line = f.readline()
f.close()

with open("/Users/hyeonghun/Desktop/swsss2023/swsss2023/day_02/omni_test.lst") as f:
    line = f.readline()
    line = f.readline()
    
with open("/Users/hyeonghun/Desktop/swsss2023/swsss2023/day_02/omni_test.lst") as f:
    for line in f:
        print(line)
        
nLines = 3
with open("/Users/hyeonghun/Desktop/swsss2023/swsss2023/day_02/omni_test.lst") as f:
    
    year = []
    day = []
    hour = []
    minute = []
    symh = []
    times = []
    
    # part 1:
    # skip lines 1-3
    for i in range(nLines):
        print(f.readline())
        
    # part 2:
    # read line 4: read in variables line and 
    # convert to variable names
    header = f.readline()
    vars = header.split()
    print(vars)
    
    # Part 3:
    # read in data line by line, convert to numerical values  
    for line in f:
        tmp = line.split()
        
      
        # print substrings for each line
        # ---- add codes here ----------------
        year.append(int(tmp[0]))
        day.append(int(tmp[1]))
        hour.append(int(tmp[2]))
        minute.append(int(tmp[3]))
        symh.append(float(tmp[4]))
        #------------------------------------
        
        datetime1 = dt.datetime(int(tmp[0]), 1, 1, int(tmp[2]), int(tmp[3])) + dt.timedelta(days = int(tmp[1])-1)
    
    print(datetime1)
    
    times.append(datetime1)
    
# =============================================================================
# print(hour)
# 
# print(minute)
# =============================================================================

time1 = dt.datetime(2013,1,3,10,12,10)
time2 = dt.datetime(2013,1,1,10,12,10) + dt.timedelta(days = 2)

# =============================================================================
# print(time1)
# =============================================================================

datetime1 = dt.datetime(int(tmp[0]), 1, 1, int(tmp[2]), int(tmp[3])) + dt.timedelta(days = int(tmp[1])-1)

