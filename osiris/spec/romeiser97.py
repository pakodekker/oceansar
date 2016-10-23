""" Omnidirectional spectrum of Romeiser 97 model

    Reference:
        "An improved composite surface model for the radar backscattering cross 
        section of the ocean surface", R. Romeiser and W. Alpers (1997)
"""

import numpy as np
from scipy import interpolate
from scipy.constants import g

def romeiser97(k, U_10, fetch):
    
    # Peak wave number (38)
    k_p = g / np.sqrt(2) / U_10**2
    # A factor which describes a low roll-off and JONSWAP peaking as function of wind speed (39)
    P_L_f = 0.00195*np.exp(-(k_p/k)**2 + 0.53*np.exp(-(np.sqrt(k) - np.sqrt(k_p))**2 / (0.32 * k_p)))
        
    # Wind speed exponent (41)
    k_1 = 183.
    k_2 = 3333.
    k_3 = 33.
    k_4 = 140.
    k_5 = 220.
    beta_f = (1 - np.exp(-(k/k_1)**2)) * np.exp(-k/k_2) + (1 - np.exp(-k/k_3)) * np.exp(-((k-k_4)/k_5)**2)
    
    # Spectral shape (42)
    k_6 = 280.
    k_7 = 75.
    k_8 = 1300.
    k_9 = 8885.
    W_H_f = np.sqrt(1 + (k/k_6)**7.2) / ((1 + (k/k_7)**2.2) * (1 + np.power(k/k_8, 3.2))**2) * np.exp(-(k/k_9)**2)
    
    Sk = P_L_f * W_H_f * (U_10**beta_f) / (k**3)
        
    return Sk