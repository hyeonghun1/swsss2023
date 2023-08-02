#!/usr/bin/env python

# Author : Hyeonghun Kim
# Aug. 2. 2023

import numpy as np
import matplotlib.pyplot as plt
from tridiagonal import solve_tridiagonal


# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

def scale_height(temp, gravity, mass):
    """Returns scale height given temperature and accel. due to gravity
    """
    return k*temp/mass/gravity

if __name__ == "__main__":

    
    dx = 4

    # set x with 1 ghost cell on both sides:
    # x = np.arange(-dx, 10 + 2 * dx, dx)

    x = 100 + np.arange(-dx, 400 + 2*dx, dx)
    
    
    t_lower = 200.0
    t_upper = 1000.0

    nPts = len(x)
    

    # set default coefficients for the solver:
    a = np.zeros(nPts) + 1
    b = np.zeros(nPts) - 2
    c = np.zeros(nPts) + 1
    d = np.zeros(nPts)


    # Add a source term Q: 
    Q = np.zeros(nPts)
    Q[(x>200) & (x<400)] = 0.4
    k = 80
    dz = x[1] - x[0]
    d = -Q * dz**2 / k
    
    
    # another source
    Q_EUV = np.zeros(nPts)
       
    
    ## time dependence
    nDays = 3
    
    # hours:
    dt = 1
    
    times = np.arange(0, nDays*24, dt)
    
    lon = 0.0
    SunHeat = 0.4
    
    
    # plot:
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    
    Temperature = np.zeros([len(x), len(times)])
    
    alt_2D, time_2D = np.meshgrid(x, times)

    
    AmpDi = 10
    AmpSd = 5
    PhaseDi = np.pi/2
    PhaseSd = 3*np.pi/2
    
    # f10.7
    f10_7 = 100 + 50/(24*365) * times + 25*np.sin(times/(27*24) * 2*np.pi)
    
    # mass of N2, O2, O
    m = [28*1.67e-27, 32*1.67e-27, 16*1.67e-27]

    # n
    n0_N2 = 1 * 10**19
    n0_O2 = 0.3* 10**19
    n0_O = 1* 10**18

    radius_e = 6370  # km
    
    for i, hour in enumerate(times):
        
        SunHeat = f10_7[i] * 0.4/100
        
        ut = hour % 24
        LocalTime = lon/15 + ut
        
        fac = -np.cos(LocalTime/24 * 2*np.pi)
        if fac < 0:
            fac = 0
        
        Q_EUV[(x>200) & (x<400)] = SunHeat * fac
        Qback = Q
        Q_new = -(Qback + Q_EUV) * dz**2 / k
        
        d = Q_new
        
        t_lower = 200 + AmpDi * np.sin(LocalTime/24*2*np.pi + PhaseDi)
        + AmpSd * np.sin(LocalTime/24*2*np.pi*2 + PhaseSd)

        
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


        # ax.plot(x,t)
        # ax.set_xlabel('Altitude', fontsize = 15)
        # ax.set_ylabel('Temperature', fontsize = 15)
        # ax.set_title('Heat Conduction throughout the altitude', fontsize = 18)



    plotfile = 'conduction_v1.png'
    print('writing : ',plotfile)    
    fig.savefig(plotfile)
    plt.close()
    
    
    # plt.contourf(time_2D/24, alt_2D, Temperature.T)
    plt.contourf(time_2D/24, alt_2D, Temperature.T, levels=25)
    plt.xlabel('Time (days)')
    plt.ylabel('Altitude')
    plt.xlim([0,3])
    plt.colorbar(label = "Temperature [k]")
    plt.title('Temperature over time and altitude')
    
    
    # plt.plot(x, n)

  

    # ax.plot(x, t)
    # plt.xlabel('Altitude', fontsize = 15)
    # plt.ylabel('Temperature', fontsize = 15)
    # plt.title('Heat Conduction throughout the altitude', fontsize = 18)


    
    
    
