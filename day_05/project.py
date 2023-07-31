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

df_all = []
JB2008_hour_all = []

for i in range(len(Champ_data)):
    df = pd.read_csv(Champ_data[i], delim_whitespace=True, header=None, skiprows=1)
    df.columns = header_label
    df.head()    
    
    hourly_index = np.where(df['GPS Time (sec)'] % 3600 == 0)[0]
    
    df_hour = df.iloc[hourly_index]
    
    df_all.append(df_hour)
    
    JB2008_hour = df_hour.loc[:,["Local Solar Time (hours)", "Geodetic Altitude (km)", "Geodetic Latitude (deg)"]]
    
    JB2008_hour_all.append(JB2008_hour)


#%%
# Import required packages
from scipy.io import loadmat


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
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008, nofLat_JB2008,
                                               nofAlt_JB2008, 8760), order='F')

#%%
"""
Data Interpolation (3D)

"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator


JB2008_interpolated = []

for day in range(len(JB2008_hour_all)):
    interp_day = []
    for hour in range(len(JB2008_hour_all[day])):  
        # Generate 3D-Interpolant every hour
        JB2008_interp = RegularGridInterpolator((localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008),
                                                 JB2008_dens_reshaped[:,:,:,day*24 + hour],
                                                 bounds_error = False, fill_value = None)
    
        interp_day.append(JB2008_interp((JB2008_hour_all[day].iloc[hour]["Local Solar Time (hours)"],
                                    JB2008_hour_all[day].iloc[hour]["Geodetic Latitude (deg)"],
                                    JB2008_hour_all[day].iloc[hour]["Geodetic Altitude (km)"] )))
        
    JB2008_interpolated.append(np.array(interp_day))
    
JB2008_interpolated = np.array(JB2008_interpolated)


# reshape a matrix to a vector
JB2008_50 = np.reshape(JB2008_interpolated, 50*24)


#%% TIE-GCM

# Import required packages
from scipy.io import loadmat
import h5py

loaded_data_tiegcm = loaded_data = h5py.File('/Users/hyeonghun/Desktop/Data/TIEGCM/2002_TIEGCM_density.mat')

# This is a HDF5 dataset object, some similarity with a dictionary
print('Key within dataset:',list(loaded_data.keys()))

tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

# We will be using the same time index as before.
time_array_tiegcm = time_array_JB2008

tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')


#%%
"""
Data Interpolation (3D)

"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator


tiegcm_interpolated = []

for day in range(len(tiegcm_hour_all)):
    inter = []
    for hour in range(len(tiegcm_hour_all[day])):  
        # Generate 3D-Interpolant every hour
        tiegcm_interp = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm),
                                                 tiegcm_dens_reshaped[:,:,:,day*24 + hour],
                                                 bounds_error = False, fill_value = None)
    
        inter.append(tiegcm_interp((tiegcm_hour_all[day].iloc[hour]["Local Solar Time (hours)"],
                                    tiegcm_hour_all[day].iloc[hour]["Geodetic Latitude (deg)"],
                                    tiegcm_hour_all[day].iloc[hour]["Geodetic Altitude (km)"] )))
        
    tiegcm_interpolated.append(np.array(inter))
    
tiegcm_interpolated = np.array(tiegcm_interpolated)


# reshape a matrix to a vector
tiegcm_50 = np.reshape(tiegcm_interpolated, 50*24)






#%% plot
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, figsize = (10, 4))
cs = plt.plot(JB2008_50)
axs.set_xlabel('Days', fontsize = 13)
axs.set_ylabel('JB2008 density', fontsize = 13)
axs.set_title('Interpolated JB2008 for 50 days', fontsize = 15)



fig, axs = plt.subplots(1, figsize = (10, 4))
cs = plt.plot(tiegcm_50)
axs.set_xlabel('Days', fontsize = 13)
axs.set_ylabel('JB2008 density', fontsize = 13)
axs.set_title('Interpolated JB2008 for 50 days', fontsize = 15)



