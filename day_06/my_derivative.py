#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math

# ----------------------------------------------------------------------
# Take first derivative of a function
# ----------------------------------------------------------------------

def first_derivative(f, x):

    """ Function that takes the first derivative

    Parameters
    ----------
    f - values of a function that is dependent on x -> f(x)
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    dfdx - the first derivative of f(x)

    Notes
    -----
    take the first derivative of f(x) here

    """

    nPts = len(f)
    
    dfdx = np.zeros(nPts)
    
    dx = x[1] - x[0]

    # do calculation here - need 3 statements:
    
    #  1. central region (using spans, like dfdx(1:nPts-2) = ...)
    dfdx[1 : nPts-2] = (f[2 : nPts-1] - f[0 : nPts-3])/(2*dx)
    
    #  2. left boundary ( fwd second order scheme )
    dfdx[0] = (-3*f[0] + 4*f[1] - f[2])/(2*dx) ;
    
    #  3. right boundary ( bwd second order scheme )
    dfdx[nPts-1] = (3*f[nPts-1] - 4*f[nPts-2] + f[nPts-3])/(2*dx);

    return dfdx

# ----------------------------------------------------------------------
# Take second derivative of a function
# ----------------------------------------------------------------------

def second_derivative(f, x):

    """ Function that takes the second derivative

    Parameters
    ----------
    f - values of a function that is dependent on x -> f(x)
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    d2fdx2 - the second derivative of f(x)

    Notes
    -----
    take the second derivative of f(x) here

    """

    nPts = len(f)
    
    d2fdx2 = np.zeros(nPts)

    dx = x[1] - x[0]

    # do calculation here - need 3 statements:
    
    #  1. central region (using spans, like dfdx(1:nPts-2) = ...)
    d2fdx2[1 : nPts-2] = (f[2 : nPts-1] + f[0 : nPts-3] - 2*f[1 : nPts-2])/(dx**2)
    
    #  2. left boundary ( forward second order scheme )
    d2fdx2[0] = (f[0] - 2*f[1] + f[2])/(dx**2)
    
    #  3. right boundary ( backward second order scheme )
    d2fdx2[nPts-1] = (f[nPts-1] - 2*f[nPts-2] + f[nPts-3])/(dx**2)

    return d2fdx2

# ----------------------------------------------------------------------
# Get the analytic solution to f(x), dfdx(x) and d2fdx2(x)
# ----------------------------------------------------------------------

def analytic(x):

    """ Function that gets analytic solutions

    Parameters
    ----------
    x - the location of the point at which f(x) is evaluated

    Returns
    -------
    f - the function evaluated at x
    dfdx - the first derivative of f(x)
    d2fdx2 - the second derivative of f(x)

    Notes
    -----
    These are analytic solutions!

    """

    f = np.exp(-x) + np.sin(3* x**2)
    dfdx = -np.exp(-x) + np.cos(3* x**2)*(6*x)
    d2fdx2 = np.exp(-x) - np.sin(3 * x**2) * (6*x)**2 + 6*np.cos(3* x**2)

    return f, dfdx, d2fdx2

# ----------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # get figures:
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    # define dx:
    dx = np.pi / 1000
    
    # arange doesn't include last point, so add explicitely:
    x = np.arange(-2.0 * np.pi, 2.0 * np.pi + dx, dx)

    # get analytic solutions:
    f, a_dfdx, a_d2fdx2 = analytic(x)

    # get numeric first derivative:
    n_dfdx = first_derivative(f, x)

    # get numeric first derivative:
    n_d2fdx2 = second_derivative(f, x)


    # plot:
    ax1.plot(x, f)
    ax1.set_title('Original function')

    # plot first derivatives:
    error = np.sum(np.abs(n_dfdx - a_dfdx)) / len(n_dfdx)
    sError = ' (Err: %5.1f)' % error
    ax2.plot(x, a_dfdx, color = 'black', label = 'Analytic')
    ax2.plot(x, n_dfdx, color = 'red', label = 'Numeric'+ sError)
    ax2.set_title('First derivative')
    ax2.scatter(x, n_dfdx, color = 'red')
    ax2.legend()

    # plot second derivatives:
    error1 = np.sum(np.abs(n_d2fdx2 - a_d2fdx2)) / len(n_d2fdx2)
    sError1 = ' (Err: %5.1f)' % error1
    ax3.plot(x, a_d2fdx2, color = 'black', label = 'Analytic')
    ax3.plot(x, n_d2fdx2, color = 'red', label = 'Numeric'+ sError1)
    ax3.set_title('Second derivative')
    ax3.scatter(x, n_d2fdx2, color = 'red')
    ax3.legend()
        
    # plotfile = 'plot.png'
    # print('writing : ',plotfile)    
    # fig.savefig(plotfile)
    # plt.close()
    
    #%% Assignment 2: error plot
    npts = [10, 20, 40, 80, 160, 320, 640, 1280]
    
    first_error = np.zeros(len(npts))
    second_error = np.zeros(len(npts))
    
    for i_pts in range(len(npts)):
    
        # define dx:
        dx = np.pi / npts[i_pts]
        
        # arange doesn't include last point, so add explicitely:
        x = np.arange(-2.0 * np.pi, 2.0 * np.pi + dx, dx)
    
        # get analytic solutions:
        f, a_dfdx, a_d2fdx2 = analytic(x)
    
        # get numeric first derivative:
        n_dfdx = first_derivative(f, x)
    
        # get numeric first derivative:
        n_d2fdx2 = second_derivative(f, x)
    
        # error
        first_error[i_pts] = np.sum(np.abs(n_dfdx - a_dfdx) / np.abs(a_dfdx) ) / len(n_dfdx)
        
        second_error[i_pts] = np.sum(np.abs(n_d2fdx2 - a_d2fdx2) / np.abs(a_dfdx) ) / len(n_d2fdx2)
        
       
        
    plt.semilogy(npts, first_error)
    plt.xlabel('Number of points')
    plt.ylabel('Error')

    plt.semilogy(npts, second_error)
    plt.xlabel('Number of points')
    plt.ylabel('Error')    
    
    plt.title('Sum of relative errors for derivatives')
    plt.grid()
    plt.legend(['First derivative', 'Second derivative'])
    
    # ax1.plot(x, f)
    # ax1.set_title('Original function')

    # # plot first derivatives:
    # error = np.sum(np.abs(n_dfdx - a_dfdx)) / len(n_dfdx)
    # sError = ' (Err: %5.1f)' % error
    # ax2.plot(x, a_dfdx, color = 'black', label = 'Analytic')
    # ax2.plot(x, n_dfdx, color = 'red', label = 'Numeric'+ sError)
    # ax2.set_title('First derivative')
    # ax2.scatter(x, n_dfdx, color = 'red')
    # ax2.legend()

    # # plot second derivatives:
    # error1 = np.sum(np.abs(n_d2fdx2 - a_d2fdx2)) / len(n_d2fdx2)
    # sError1 = ' (Err: %5.1f)' % error1
    # ax3.plot(x, a_d2fdx2, color = 'black', label = 'Analytic')
    # ax3.plot(x, n_d2fdx2, color = 'red', label = 'Numeric'+ sError1)
    # ax3.set_title('Second derivative')
    # ax3.scatter(x, n_d2fdx2, color = 'red')
    # ax3.legend()
    
