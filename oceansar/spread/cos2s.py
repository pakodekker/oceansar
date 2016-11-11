"""
Cosine raised to the 2s power directional wave function. The model
is based on that presented in in Hasselmann et al. (1980), but has
been translated to the wave number domain. The function is
normalized so that the integral from -pi to pi is 1. The function
depends on the model used for the omnidirectional spectrum to
calculate wave number at the peak of the spectrum. Here, we use the
Hasselmann (1976) spectrum (JONSWAP spectrum).
"""

import numpy as np

def cos2s(k, theta, U_10, fetch):
    
    G = 1./np.pi*np.cos(theta)**2
    G[(theta < -np.pi/2.) or (theta > np.pi/2.)] = 0.

    return G

