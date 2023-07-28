#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Creating a random numpy array
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)
print(data_arr.shape)

#%%
"""
TODO: Writing and reading numpy file
"""
# Save the data_arr variable into a .npy file
np.save('test_np_save.npy', data_arr)

# Load data from a .npy file
data_arr_loaded = np.load('test_np_save.npy')

# Verification that the loaded data matches the initial data exactly
print(np.equal(data_arr, data_arr_loaded))
print(data_arr == data_arr_loaded)

# Verify that the loaded data matches the initial data

#%%
"""
TODO: Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz', data_arr, data_arr2)

# Load the numpy zip file
npzfile = np.load('test_savez.npz')

print(npzfile.files)

#%%

# To inspect the name of the variables within the npzfile
print('Variable names within this file:', sorted(npzfile.files))

# We will then be able to use the variable name as a key to access the data.
print(npzfile['arr_0'])

# Verification that the loaded data matches the initial data exactly
print((data_arr==npzfile['arr_0']).all())
print((data_arr2==npzfile['arr_1']).all())

#%%
"""
Error and exception
"""
# =============================================================================
# np.equal(data_arr,npzfile)
# We can compare a single file with a single file
# =============================================================================

# Exception handling, can be used with assertion as well
try:
    # Python will try to execute any code here, and if there is an exception 
    # skip to below 
    print(np.equal(data_arr,npzfile).all())
except:
    # Execute this code when there is an exception (unable to run code in try)
    print("The codes in try returned an error.")
    print(np.equal(data_arr, npzfile['arr_0']).all())
    
#%%
"""
TODO: Error solving 1
"""
# What is wrong with the following line? 
np.equal(data_arr, data_arr2)

# -> They have the different sizes of matrices, so can't compare each other

#%%
"""
TODO: Error solving 2
"""
# What is wrong with the following line? 
np.equal(data_arr2, npzfile['data_arr2'])

#%%
"""
TODO: Error solving 3
"""
# What is wrong with the following line? 
numpy.equal(data_arr2, npzfile['arr_1'])


#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
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

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

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
time_array_JB2008 = np.linspace(0,8759,5, dtype = int)

# For the dataset that we will be working with today, you will need to reshape 
# them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,
                                               nofAlt_JB2008,8760), order='F') # Fortran-like index order

#%%
"""
TODO: Plot the atmospheric density for 400 KM for the first time index in
      time_array_JB2008 (time_array_JB2008[0]).
"""

import matplotlib.pyplot as plt

# Look for data that correspond to an altitude of 400 KM
alt = 400
hi = np.where(altitudes_JB2008 == alt) # find the index of 400km of altitude

# Create a canvas to plot our data on. Here we are using a subplot for the plots.
fig, axs = plt.subplots(1, figsize=(15, 5), sharex=True)

ik =0
axs.contourf(localSolarTimes_JB2008, latitudes_JB2008,
             JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
axs.set_xlabel("Local Solar Time", fontsize=18) 
axs.set_ylabel("Latitudes", fontsize=18)
axs.set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)

axs.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cs,ax=axs)
cbar.ax.set_ylabel('Density')

plt.show()
#%%
"""
TODO: Plot the atmospheric density for 300 KM for all time indexes in
      time_array_JB2008
"""
import matplotlib.pyplot as plt

fig, axs = plt.subplots((5), figsize=(15, 10*2), sharex=True)


for ik in range(5):
    # print(ik)
    tmp = axs[ik].contourf(localSolarTimes_JB2008, latitudes_JB2008,
                 JB2008_dens_reshaped[:,:,hi,time_array_JB2008[ik]].squeeze().T)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].set_title('JB2008 density at 400 km, t = {} hrs'.format(time_array_JB2008[ik]), fontsize=18)
    
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
        
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(tmp,ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18) 




#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. 
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]
print('The dimension of the data are as followed (local solar time,latitude,altitude):' , dens_data_feb1.shape)

# Method 1: For loop
mean_dens = []

for altitude in range(len(altitudes_JB2008)):
    mean_dens.append(np.mean(dens_data_feb1[:,:,altitude].squeeze()))
    
mean_dens = np.array(mean_dens)

plt.semilogy(altitudes_JB2008, mean_dens)
plt.xlabel('Altitude')
plt.ylabel('Density')
plt.title('Mean density for each altitude')
plt.grid()
plt.show()

# Method 2: List comprehension
mean_dens2 = [np.mean(dens_data_feb1[:,:,ii]) for ii in range(len(altitudes_JB2008))] # mean_desn2 : list
mean_dens2 = np.array(mean_dens2)
plt.semilogy(altitudes_JB2008, mean_dens2)
plt.xlabel('Altitude')
plt.ylabel('Density')
plt.title('Mean density for each altitude')
plt.grid()
plt.show()

# =============================================================================
# fig, axs = plt.subplots(2, 1)
# axs[0, 0].plot(altitudes_JB2008, mean_dens)
# axs[0, 0].set_title('Using for loop')
# axs[1, 0].plot(altitudes_JB2008, mean_dens2, 'tab:orange')
# axs[1, 0].set_title('Using list comprehension')
# =============================================================================


#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density 
field at 310km

"""
# Import required packages
import h5py
loaded_data = h5py.File('/Users/hyeonghun/Desktop/Data/TIEGCM/2002_TIEGCM_density.mat')

# This is a HDF5 dataset object, some similarity with a dictionary
print('Key within dataset:',list(loaded_data.keys()))


#%%
"""
TODO: Plot the atmospheric density for 310 KM for all time indexes in
      time_array_tiegcm
"""
tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

# We will be using the same time index as before.
time_array_tiegcm = time_array_JB2008

#%%
# Each data correspond to the density at a point in 3D space. 
# We can recover the density field by reshaping the array.
# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')

print(JB2008_dens_reshaped.shape, tiegcm_dens_reshaped.shape)

#%%
import matplotlib.pyplot as plt

fig, axs = plt.subplots((5), figsize=(15, 10*2), sharex=True)

#%matplotlib qt
for ik in range(5):
    # print(ik)
    tmp = axs[ik].contourf(localSolarTimes_tiegcm, latitudes_tiegcm,
                 tiegcm_dens_reshaped[:,:,hi,time_array_tiegcm[ik]].squeeze().T)
    axs[ik].set_ylabel("Latitudes", fontsize=18)
    axs[ik].set_title('TIEGCM density at 300 km, t = {} hrs'.format(time_array_tiegcm[ik]), fontsize=12)
    
    axs[ik].tick_params(axis = 'both', which = 'major', labelsize = 16)
        
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(tmp,ax=axs[ik])
    cbar.ax.set_ylabel('Density')

axs[ik].set_xlabel("Local Solar Time", fontsize=18) 

#%%
"""
Assignment 1.5

Can you plot the mean density for each altitude at February 1st, 2002 for both 
models (JB2008 and TIE-GCM) on the same plot?
"""

# First identidy the time index that corresponds to  February 1st, 2002. 
# Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1_tiegcm = tiegcm_dens_reshaped[:,:,:,time_index]
print('The dimension of the data are as followed (local solar time,latitude,altitude):' , dens_data_feb1_tie.shape)

#%%
# List comprehension
mean_dens_tiegcm = [np.mean(dens_data_feb1_tiegcm[:,:,ii]) for ii in range(len(altitudes_tiegcm))]
mean_dens_tiegcm = np.array(mean_dens_tiegcm)

plt.semilogy(altitudes_tiegcm, mean_dens_tiegcm)
plt.xlabel('Altitude')
plt.ylabel('Density')
plt.title('Mean density for each altitude')
plt.grid()
plt.show()

#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)

# Generate 1D interpolant function
interp_func_1D = interpolate.interp1d(x, y)  # linear interpolation

# Let's select some new points
xnew = np.arange(0, 9, 0.1)
# use interpolation function returned by `interp1d`
ynew = interp_func_1D(xnew) 

print(xnew)
print(ynew)


#%%

interp_func_1D_cubic = interpolate.interp1d(x, y, kind='cubic')

# use interpolation function returned by `interp1d`
ycubic = interp_func_1D_cubic(xnew) 

interp_func_1D_quadratic = interpolate.interp1d(x, y,kind='quadratic')

# use interpolation function returned by `interp1d`
yquadratic = interp_func_1D_quadratic(xnew) 

print(xnew)
print(ycubic)
print(yquadratic)

#%% plot

plt.subplots(1, figsize=(10, 6))
plt.plot(x, y, 'o', xnew, ynew, '*',xnew, ycubic, '--',xnew, yquadratic, '--',linewidth = 2)
plt.legend(['Inital Points','Interpolated line-linear','Interpolated line-cubic','Interpolated line-quadratic'], fontsize = 16)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.title('1D interpolation', fontsize = 18)
plt.grid()
plt.tick_params(axis = 'both', which = 'major', labelsize = 16)



#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)

interpolated_function_1 = RegularGridInterpolator((x, y, z), sample_data)

# Say we are interested in the points [[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]]
pts = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1]])
print('Using 3D-interpolation method:',interpolated_function_1(pts)) 
print('From true function:',function_1(pts[:,0],pts[:,1],pts[:,2]))

#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment"} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)


#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization 
grid.

Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on 
February 1st, 2002, with the discretized grid used for the JB2008 
((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""

print(altitudes_JB2008)
print(altitudes_tiegcm)
print(localSolarTimes_JB2008)
print(localSolarTimes_tiegcm)
# format of LocalST, latitude, altitude

time_index = 31*24

# Generate 3D-Interpolant (interpolating function)
tiegcm_function = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm),
                                          tiegcm_dens_reshaped[:,:,:,time_index],
                                          bounds_error = False, fill_value = None)


tiegcm_on_JB2008 = np.zeros((len(localSolarTimes_JB2008), len(latitudes_JB2008)))
for lst_i in range(len(localSolarTimes_JB2008)):
    for lat_j in range(len(latitudes_JB2008)):
        tiegcm_on_JB2008[lst_i, lat_j] = tiegcm_function((localSolarTimes_JB2008[lst_i],
                                                          latitudes_JB2008[lat_j], 400))
#%% Using meshgrid

import numpy as np
x = np.linspace(0, 1, 3)
y = np.linspace(0, 1, 2)
xy = np.meshgrid(x, y)
print(np.array(xy))

#%% Generate the meshgrid on JB2008 grid at altitude 400km

xyz_grid = np.meshgrid(localSolarTimes_JB2008, latitudes_JB2008, 400)
# xyz_grid = list of size 3 x 20(lat) x 24(lst) x 1

xyz_grid_squeeze = np.array(xyz_grid).squeeze()
# this is an array of size 3 x 20(lst) x 24(lat)
# 20: x meshgrid,  24: y meshgrid
# lst = y axis of 24, lat = x axis of 20

print(xyz_grid_squeeze)

xyz_grid_squeeze_T = xyz_grid_squeeze.T 
# size of 24 x 20 x 3

# Evaluate the density using interpolant
tiegcm_dens_400 = tiegcm_function(xyz_grid_squeeze_T)

#%% plot

fig, axs = plt.subplots(2, figsize = (8, 5), sharex = True)  # sharex = share the x axis for subplots

# tiegcm_on_JB2008 is tiegcm density field at 400km on Feb. 1st, 2002, on the discretized JB2008 grid

cf1 = axs[0].contourf(localSolarTimes_JB2008, latitudes_JB2008, tiegcm_on_JB2008.T)
axs[0].set_title('TIE-GCM density on JB2008 grid at 400 km, t = {} hrs'.format(time_index), fontsize=12)
axs[0].set_ylabel("Latitudes", fontsize=15)
    
# Colorbar
cbar = fig.colorbar(cf1, ax = axs[0])
cbar.ax.set_ylabel('Density')


alt = 400
hi = np.where(altitudes_JB2008 == alt)

cf2 = axs[1].contourf(localSolarTimes_JB2008, latitudes_JB2008, JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)
axs[1].set_title('JB2008 density on JB2008 grid at 400 km, t = {} hrs'.format(time_index), fontsize=12)
axs[1].set_ylabel("Latitudes", fontsize=15)

axs[1].set_xlabel("Local Solar Time", fontsize = 15)    
    
cbar = fig.colorbar(cf2, ax = axs[1])
cbar.ax.set_ylabel('Density')



#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this 
difference in a contour plot.
"""

density_diff = (tiegcm_on_JB2008.T - JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)

fig, axs = plt.subplots(1, figsize = (10, 4))
cs1 = plt.contourf(localSolarTimes_JB2008, latitudes_JB2008, density_diff)
axs.set_xlabel('Local Solar Time', fontsize = 12)
axs.set_ylabel('Latitudes', fontsize = 12)
axs.set_title('Density difference between TIEGCM and JB2008 at 400km, t = {} hrs'.format(time_index), fontsize = 12)

cbar = fig.colorbar(cs, ax = axs)
cbar.ax.set_ylabel('Density')

#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in 
terms of absolute percentage difference/error (APE). Let's plot the APE 
for this scenario.

APE = abs(tiegcm_dens-JB2008_dens)/tiegcm_dens
"""

APE = abs(density_diff / tiegcm_on_JB2008.T)

fig, axs = plt.subplots(1, figsize = (10, 4))
cs2 = plt.contourf(localSolarTimes_JB2008, latitudes_JB2008, APE)
axs.set_xlabel('Local Solar Time', fontsize = 12)
axs.set_ylabel('Latitudes', fontsize = 12)
axs.set_title('APE between TIEGCM and JB2008 at 400km, t = {} hrs'.format(time_index),
              fontsize = 12)
cbar = fig.colorbar(cs2, ax = axs)
cbar.ax.set_ylabel('Density')


