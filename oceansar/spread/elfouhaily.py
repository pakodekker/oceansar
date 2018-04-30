"""
Implementation of the directional function to construct a
directional wave spectrum, following Elfouhaily et al.

Elfouhaily T., Chapron B., and Katsaros K. (1997). "A unified
directional spectrum for long and short wind driven waves"
J. Geophys. Res. 102 15.781-96

LOG:
2011-08-26 Gordon Farquharson: Removed an extra factor of 2. the code that implements from Equation 49.
2013-01-25 Paco Lopez Dekker: Renormalized function so that the integral in theta gives 1 (instead of 0.5)
"""

import numpy as np
from scipy.constants import g
import numexpr as ne

rho = 1000.   # Density of water in kg/m^3
S = 0.072     # Surface tension of water in N/m
X_0 = 22e3    # Dimensionless fetch

def elfouhaily(k, theta, U_10, fetch):
    # Eq. 3 (below)
    k_0 = g/U_10**2 
    # Eq. 4 (below)
    X = k_0*fetch
    # Eq. 37
    Omega_c = 0.84*np.tanh((X/X_0)**(0.4))**(-0.75)
    cK = np.sqrt(g/k + S/rho*k)

    # Eq. 3 (below)
    k_p = k_0*Omega_c**2
    cK_p = np.sqrt(g/k_p + S/rho*k_p)

    # Eq. 24
    k_m = np.sqrt(rho*g/S)
    cK_m = np.sqrt(g/k_m + S/rho*k_m)

    # (McDaniel, 2001, above Equation 3.9)
    C_10 = (0.8 + 0.065*U_10) * 1e-3
    # Eq. 61
    ustar = np.sqrt(C_10)*U_10

    # Eq. 59
    a_0 = np.log(2.)/4.
    a_p = 4.
    a_m = 0.13*ustar/cK_m

    # Eq. 57
    Delta = np.tanh(a_0 + a_p*(cK/cK_p)**2.5 + a_m*(cK_m/cK)**2.5)
    # Eq. 49
    G = np.where((theta > -np.pi/2.) & (theta < np.pi/2.), (1. + Delta*np.cos(2.*theta))/(np.pi), 0)

    return G

