"""
Implementation of the directional function to construct the Banner
directional function, following McDaniel's paper.

McDaniel, S. T. (2001): Small-slope predictions of microwave
backscatter from the sea surface, Waves in Random Media, 11:3, 343-360.

Banner, M. L., (1990). Equilibrium spectra of wind waves. 
J. Phys. Oceanogr., 20, 966-84.
"""
    
import numpy as np
from scipy.constants import g

X_0 = 22e3     # Dimensionless fetch
        
def banner(k, theta, U_10, fetch):

    k_0 = g/U_10**2 
    X = k_0*fetch
    Omega_c = 0.84*np.tanh((X/X_0)**(0.4))**(-0.75)
    k_p = k_0*Omega_c**2
  
    # Eq. 4.4
    beta = np.where(k < 2.56*k_p,
                    2.28*(k/k_p)**(-0.65),
                    np.exp(-0.921 + 1.933*(k/k_p)**(-0.567)))

    # Eq. 4.5
    G_n = 0.5*beta/np.tanh(np.pi*beta)
    
    # TODO: Check if this is necessary
    wtheta = np.angle(np.exp(1j*theta))
    abstheta = np.abs(wtheta + np.pi)
    abstheta = np.where(abstheta > np.pi, np.abs(wtheta - np.pi), abstheta)

    # Eq. 4.3
    G = G_n*((1./np.cosh(beta*wtheta))**2 + 
             (1./np.cosh(beta*abstheta))**2)/2.

    return G
