'''
Wave number spectrum based on the frequency spectrum presented 
in Hasselmann et al. (1976). The dispersion relation used in the 
transformation is :math:`\omega = \sqrt{gk}`, so it is only valid 
for low wave numbers and deep water.

Reference:
    Hasselmann, K., W. Sell, D. B. Ross, P. Mueller, 1976: "A Parametric
    Wave Prediction Model". J. Phys. Oceanogr., 6, 200-228.

    Hasselmann, D. E., M. Dunckel, J. A. Ewing, 1980: "Directional Wave
    Spectra Observed during JONSWAP 1973". J. Phys. Oceanogr., 10, 1264-1280
'''

import numpy as np
from scipy.constants import g

def jonswap(k, U_10, fetch):

    # P. 202
    xi = g*fetch/U_10**2

    # P. 203, Eqs. 2.2, 2.3
    k_m = (2.*np.pi*3.5)**2*g/U_10**2*xi**(-0.66)
    alpha = 0.076*xi**(-0.22)
    # P. 204
    gamma = 3.3

    sigma = np.where(k > k_m, 0.09, 0.07)

    # P. 202, Eq. 2.1
    r = np.exp(-(np.sqrt(k/k_m)-1.)**2/2./sigma**2)
    Sk = 0.5*alpha*k**(-3)*np.exp(-5/4*(k_m/k)**2)*gamma**r

    return Sk 
