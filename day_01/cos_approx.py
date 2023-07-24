#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Hyeonghun Kim'
__email__ = 'hyk049@ucsd.edu'

from math import factorial
from math import pi


def cos_approx(x, accuracy=10):
    
    n = range(accuracy + 1);
    
# =============================================================================
#     cosine = sum( [(-1)**n / factorial(2*n) * x**(2*n)] )
# =============================================================================
    stack = 0
    
    for elem in n:
        stack = stack + (-1)**elem  / factorial(2*elem) * x**(2*elem)
        
    return stack



# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
