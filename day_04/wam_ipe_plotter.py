#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:34:41 2023

@author: hyeonghun
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

def plot_var(dataset, variable, figsize = (12,6)):
    
    var = dataset[variable][:]
    
    lon = dataset['lon'][:]
    lat = dataset['lat'][:]
    
    unit = dataset[variable].units
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figsize)
    

    cs = ax.pcolormesh(lon, lat, var)
    ax.set_xlabel('Longitute [degrees_east]', fontsize = 12)
    ax.set_ylabel('Latitudes [degrees_north]', fontsize = 12)
    ax.set_title('{}'.format(variable), fontsize = 15)
    
    cbar = fig.colorbar(cs, ax = ax)
    cbar.ax.set_ylabel('{}'.format(unit), fontsize = 12)


    return fig, ax



# save to png file
def save2png(infilename, var_name):
    
    dataset = nc.Dataset(infilename)
    
    fig, ax = plot_var(dataset, var_name)
    
    outfilename = infilename + var_name + '.png'
    
    png = fig.savefig(outfilename)
    
    # We don't need to return anything to save a png file
    
    return png

    
    
    
    
    
    