#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:35:14 2023

@author: hyeonghun
"""

from wam_ipe_plotter import plot_var, save2png
import sys 
import netCDF4 as nc

if __name__ == '__main__':
    
    filename = sys.argv[1]
    
    # '/Users/hyeonghun/Desktop/wfs.t06z.20230726_10/wfs.t06z.ipe05.20230726_105500.nc'
    
    var_name = 'HmF2'
    
    infilename = filename
    
    png = save2png(infilename, var_name)

    command_arguments = sys.argv[1:]
    
    for ii in range(len(command_arguments)):
        var_name = 'tec'
        
        infilename = sys.argv[1:ii]
        
        png = save2png(infilename, var_name)
    
    
    


