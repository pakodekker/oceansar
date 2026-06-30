"""
Elfouhaily et al. Omnidirectional Spectrum

Reference:
    Elfouhaily T., Chapron B., and Katsaros K. (1997). "A unified
    directional spectrum for long and short wind driven waves"
"""

import warnings

import numpy as np
from scipy.constants import g


rho = 1000.    # Density of water in kg/m^3
S = 0.072      # Surface tension of water in N/m
X_0 = 22e3     # Dimensionless fetch


def __gamma_function(Omega_c):
    # Eq. 3 (below)
    if (Omega_c > 0.84) and (Omega_c < 1.0):
        gamma = 1.7
    elif (Omega_c > 1.0) and (Omega_c < 5.0):
        gamma = 1.7 + 6.*np.log10(Omega_c)
    else:
        warnings.warn('Omega_c is out of range. Returning  value for 5.0', RuntimeWarning)
        gamma = 1.7 + 6.*np.log10(5.0)

    return gamma


def __alpha_m_function(ustar, c_m):
    # Eq. 44
    if ustar < c_m:
        alpha_m = 1e-2*(1. + np.log(ustar/c_m))
    else:
        alpha_m = 1e-2*(1. + 3.*np.log(ustar/c_m))

    return alpha_m


def elfouhaily(k, U_10, fetch, return_components='False'):
    # Calculated variables
    k_m = 2*np.pi/0.017
    # Eq. 3 (below)
    k_0 = g/U_10**2
    # Eq. 4 (below)
    X = k_0*fetch
    # Eq. 37: Inverse wave age
    Omega_c = 0.84*np.tanh((X/X_0)**(0.4))**(-0.75)
    # Wave phase speed (assumes deep water)
    c = np.sqrt(g/k + S/rho*k)

    # B_l: Long-wave curvature spectrum
    # Note that in contrast to Elfouhaily's paper, the L_pm factor
    # is applied to the both B_l and B_h.

    # Eq. 3 (below)
    k_p = k_0*Omega_c**2
    # Phase speed at the spectral peak
    c_p = np.sqrt(g/k_p + S/rho*k_p)
    # Eq. 32 (above): Inverse wave age parameter (dimensionless)
    Omega = U_10/c_p
    # Eq. 34
    alpha_p = 6e-3*np.sqrt(Omega)

    # Eq. 3 (below)
    sigma = 0.08 * (1. + 4.*Omega_c**(-3.))
    Gamma = np.exp(-(np.sqrt(k/k_p)-1.)**2 / (2.*sigma**2))
    # Eq. 3
    J_p = __gamma_function(Omega_c)**Gamma
    # Eq. 2
    L_pm = np.exp(-5./4.*(k_p/k)**2)
    # Eq. 32
    F_p = L_pm*J_p*np.exp(-Omega/np.sqrt(10.)*(np.sqrt(k/k_p)-1.))
    # Eq. 32
    B_l = 0.5*alpha_p*c_p/c*F_p

    # B_s: Short-wave curvature spectrum
    # (McDaniel, 2001, above Equation 3.9)
    C_10 = (0.8 + 0.065*U_10) * 1e-3
    # Eq. 61: Friction velocity
    ustar = np.sqrt(C_10)*U_10

    # Eq. 41 (above)
    c_m = np.sqrt(g/k_m + S/rho*k_m)
    # Eq. 41 (above)
    alpha_m = __alpha_m_function(ustar, c_m)
    # Eq. 41 with L_pm according to McDaniel, 2001 (in text below Equation 3.9)
    F_m = L_pm*np.exp(-0.25*(k/k_m-1.)**2)
    # Eq. 40
    B_h = 0.5*alpha_m*c_m/c*F_m

    # Eq. 30 (Final spectrum)
    if return_components == 'True':
        return B_l, B_h, (B_l + B_h)/k**3
    else:
        return (B_l + B_h)/k**3

