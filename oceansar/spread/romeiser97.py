""" Spreading function of Romeiser 97 model

    Reference:
        "An improved composite surface model for the radar backscattering cross
        section of the ocean surface", R. Romeiser and W. Alpers (1997)
"""

import numpy as np


def romeiser97(k, theta, U_10, fetch):
    # Parameter 'delta' (44) [Fixed to avoid log(0)!]
    c_1 = 400.
    _1_2d2 = np.where(k == 0., np.infty,
                               0.14 + 0.5*(1 - np.exp(-k*U_10/c_1)) + 5*np.exp(2.5 - 2.6*np.log(U_10) - 1.3*np.log(k)))
    # Spreading function (43)
    # With an normalization factor so that it integrates to 1
    S = np.exp(-theta**2 * _1_2d2) / np.sqrt(np.pi / _1_2d2)

    return S