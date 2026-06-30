''' Monochromatic Swell spectrum '''

import numpy as np

def mono(k, theta, hs, wl, k_res):
    ''' Monochromatic Swell spectrum
        
        :param k: k meshgrid
        :param theta: theta meshgrid (wind direction shifted)
        :param hs: Swell RMS
        :param wl: Swell wavelength peak (standard values: 200-300m)
        :param k_res: [Kx, Ky] resolution
        
        .. note::
            Wavelength peak is adjusted to fall into nearest spectrum grid point.
    '''
    
    # Cartesian space
    kx = k*np.cos(theta)
    
    Sk = np.zeros_like(kx)
    Sk[0, np.abs(kx[0] - 2.*np.pi/wl).argmin()] = hs**2/(k_res[0]*k_res[1])
    
    return Sk

