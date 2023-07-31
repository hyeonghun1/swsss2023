#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:23:43 2023

@author: hyeonghun
"""

import os
import zipfile
import pandas as pd
import numpy as np


datalist = []
for file in os.listdir("../day_05/Champ_dens_2002/"):
    if file.endswith(".txt"):
        datalist.append(os.path.join("../day_05/Champ_dens_2002/", file))

# sort the data
Champ_data = sorted(datalist) 

assert len(Champ_data) == 50

# print the list
print(Champ_data)

# The following code reads the first file in the sortedChampdata list.
header_label = ['GPS Time (sec)','Geodetic Altitude (km)','Geodetic Latitude (deg)','Geodetic Longitude (deg)',
                'Local Solar Time (hours)','Velocity Magnitude (m/s)','Surface Temperature (K)',
                'Free Stream Temperature (K)','Yaw (rad)','Pitch (rad)','Proj_Area_Eric (m^2)','CD_Eric (~)',
                'Density_Eric (kg/m^3)','Proj_Area_New (m^2)','CD_New (~)','Density_New (kg/m^3)',
                'Density_HASDM (kg/m^3)','Density_JB2008 (kg/m^3)']


df = pd.read_csv(Champ_data[2], delim_whitespace=True, header=None, skiprows=1)
df.columns = header_label
df.head()

 
hourly_index = np.where(df['GPS Time (sec)'] % 3600 == 0)[0]
df_hour = df.iloc[hourly_index]

JB2008_hour = df_hour.loc[:,"Density_JB2008 (kg/m^3)"]


#%%
# Import required packages
from scipy.io import loadmat
import matplotlib.pyplot as plt

dir_density_Jb2008 = '/Users/hyeonghun/Desktop/Data/JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")


# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']


# Before we can visualize our density data, we first need to generate the 
# discretization grid of the density data in 3D space. We will be using 
# np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = hourly_index

# For the dataset that we will be working with today, you will need to reshape 
# them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,
                                               nofAlt_JB2008,8760), order='F')

#%%
"""
Data Interpolation (3D)

"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator


# Generate 3D-Interpolant (interpolating function)
JB2008_interp = RegularGridInterpolator((localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008),
                                          JB2008_dens_reshaped[:,:,:,time_array_JB2008],
                                          bounds_error = False, fill_value = None)


# =============================================================================
# tiegcm_on_JB2008 = np.zeros((len(localSolarTimes_JB2008), len(latitudes_JB2008)))
# for lst_i in range(len(localSolarTimes_JB2008)):
#     for lat_j in range(len(latitudes_JB2008)):
#         tiegcm_on_JB2008[lst_i, lat_j] = tiegcm_function((localSolarTimes_JB2008[lst_i],
#                                                           latitudes_JB2008[lat_j], 400))
# 
# =============================================================================









