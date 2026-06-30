''' Swell spectrum based on Durden and Vesecky (1985)
    
    References:
        A Physical Radar Cross-Section Model for a Wind-Driven 
        Sea with Swell. S. Durden et al. (1985) 
'''

import numpy as np

def gaussian(kx, ky, dir, hs, wl, sigma):
    ''' Gaussian Swell spectrum
        
        :param kx: Kx meshgrid
        :param ky: Ky meshgrid
        :param dir: Swell direction
        :param hs: Swell RMS
        :param wl: Swell wavelength peak (standard values: 200-300m)
        :param sigma: Swell deviation (k, standard values: 0.0025) 
        
        .. note::
            Wavelength peak is adjusted to fall into nearest spectrum grid point.
    '''
    # Make Kx,Ky pass trough zero
    kx -= np.min(np.abs(kx))
    ky -= np.min(np.abs(ky))
    
    # Calculate nearest peak grid points
    kx_p = kx[0, np.abs(kx[0, :] - np.cos(dir)*2.*np.pi/wl).argmin()]
    ky_p = ky[np.abs(ky[:, 0] - np.sin(dir)*2.*np.pi/wl).argmin(), 0]
    
    Sk = hs**2./(2.*np.pi*sigma**2.)*np.exp(-0.5*(((kx - kx_p)/sigma)**2 + 
                                                  ((ky - ky_p)/sigma)**2))
    
    return Sk

