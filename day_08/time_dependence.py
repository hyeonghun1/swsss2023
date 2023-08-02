#!/usr/bin/env python

# Author : Hyeonghun Kim
# Aug. 2. 2023

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":

    dx = 0.25

    # set x with 1 ghost cell on both sides:
    x = np.arange(-dx, 10 + 2 * dx, dx)

    t_lower = 200.0
    t_upper = 500.0

    nPts = len(x)
    

    # set default coefficients for the solver:
    a = np.zeros(nPts) + 1
    b = np.zeros(nPts) - 2
    c = np.zeros(nPts) + 1
    d = np.zeros(nPts)

    # Add a source term:
    Q = np.zeros(nPts)
    Q[(x>3) & (x<7)] = 100
    k = 10.0
    dz = x[1] - x[0]
    d = -Q * dz**2 / k
    
    
    # another source
    Q_EUV = np.zeros(nPts)
    
    # LocalTime = np.linspace(0, 24, 25)
    
    # for i in range(len(LocalTime)):
    #     if LocalTime[i] < 6:
    #         fac = 0
    #     elif LocalTime[i] >=6 & LocalTime[i] <=18:
    #         fac = -np.cos(LocalTime/24 * 2*np.pi)
    #     else:
    #         fac = 0
       
    
    ## time dependence
    nDays = 3
    
    # hours:
    dt = 1
    
    times = np.arange(0, nDays*24, dt)
    
    lon = 0.0
    SunHeat = 100
    
    # plot:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    
    Temperature = np.zeros([len(x), len(times)])
    
    alt_2D, time_2D = np.meshgrid(x, times)

    # ii = 0
    
    for i, hour in enumerate(times):
        ut = hour % 24
        LocalTime = lon/15 + ut
        
        fac = -np.cos(LocalTime/24 * 2*np.pi)
        if fac < 0:
            fac = 0
        
        Q_EUV[(x>3) & (x<7)] = SunHeat * fac
        Qback = Q
        Q_new = -(Qback + Q_EUV) * dz**2 / k
        
        d = Q_new
        
        # boundary conditions (bottom - fixed):
        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = t_lower
        
        # top - floating:
        a[-1] = 1
        b[-1] = -1
        c[-1] = 0
        d[-1] = 0

        # solve for Temperature:
        t = solve_tridiagonal(a, b, c, d)
        
        Temperature[: , i] = t

        # ii  = ii + 1

        ax.plot(x,t)
        ax.set_xlabel('Altitude', fontsize = 15)
        ax.set_ylabel('Temperature', fontsize = 15)
        ax.set_title('Heat Conduction throughout the altitude', fontsize = 18)



    plotfile = 'conduction_v1.png'
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.close()
    
    
    plt.contourf(time_2D/24, alt_2D*40 + 100, Temperature.T)
    plt.xlabel('Time (days)')
    plt.ylabel('Altitude')
    plt.xlim([0,3])
    plt.colorbar(label = "Temperature [k]")
    plt.title('Temperature over time and altitude')
    

  

    # ax.plot(x, t)
    # plt.xlabel('Altitude', fontsize = 15)
    # plt.ylabel('Temperature', fontsize = 15)
    # plt.title('Heat Conduction throughout the altitude', fontsize = 18)


    
    
    
