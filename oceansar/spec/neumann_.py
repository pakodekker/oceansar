'''
Implementation of Neumann wave-spectrum according to equation
8.14:15 in Kinsman's "Wind Waves".

TODO: Change to 'k' domain
'''
    
import numpy as np
from scipy.constants import g

C = 3.05                    # [m^2 s^{-5}], Kinsman (1965) p. 390
max_wavelength = 100.       # Wavelength of the longest waves [m]

def neumann(omega, theta, U_10, fetch):
    # Cutoff is gives the lowest frequency (Hz). This depends on fetch or time.
    cutoff = np.sqrt(g*2.*np.pi/max_wavelength)/(2.*np.pi)
    omega_1 = 2.*np.pi*cutoff

    res = np.where((omega > omega_1) & (np.cos(theta) > 0.),
                   C*omega**(-6)*np.exp(-2.*g**2*omega**(-2)*U_10**(-2))*(np.cos(theta))**2,
                   0.)

    return res

