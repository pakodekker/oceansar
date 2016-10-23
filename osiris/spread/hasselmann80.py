""" Speading function based on Longuet-Higggins (1963) and Hasselmann (1980)
"""

import numpy as np
from scipy import special
from scipy.constants import g

def hasselmann80(k, theta, U_10, fetch):
    
    # Haaselmann (1976) P. 202
    xi = g*fetch/U_10**2                                
    c_m = 1./(2.*np.pi)*(3.5)**(-1)*U_10*xi**0.33
    # Hasselmann (1976) Eq. 2.2
    k_p = (2.*np.pi*3.5)**2*g/U_10**2*xi**(-0.66)
    
    s = np.where(k < k_p,
                 6.97*(k/k_p)**2.03,
                 9.77*(k/k_p)**(-(0.32 + 0.725*U_10/c_m)))
    
    # TODO: Check if this is necessary
    wtheta = np.angle(np.exp(1j*theta))
    D = 2**(2*s - 1)/np.pi*special.gamma(s + 1.)**2/special.gamma(2.*s + 1.)*np.cos(wtheta/2.)**(2.*s)
    
    return D