'''
Implementation of SWOP spreading function

Nagai, K. "Computation of Refraction and Diffraction of 
Irregular Sea". Report of the Port and Harbour Research
Institute, Vol. 11, No. 2, June 1972. (in Japanese)

.. warning::
    Wind speed is defined above 5m of the surface
    
'''
    
import numpy as np
from scipy.constants import g


def swop(k, theta, U_5, fetch):
    
    k_p = g/U_5**2
    dFdK = (1/(4*np.pi))*np.sqrt(g/k)
    
    Dk = np.where(np.abs(theta) <= np.pi/2.,
                  dFdK * (1./np.pi)*(1 + (0.50 + 0.82*np.exp(-0.5*(k/k_p)**2)*np.cos(2*theta)) 
                                       + (0.32*np.exp(-0.5(k/k_p)**2)*np.cos(4*theta))), 0.)

    return Dk
