""" WAVESIM Test """

import numpy as np
import matplotlib.pyplot as plt
from osiris.surfaces import OceanSurface

from trampa import utils

def example():

    # Create ocean surface
    ocean = OceanSurface()
    ocean.init(Lx=512., Ly=512., dx=1., dy=1., cutoff_wl='auto',
               spec_model='elfouhaily', spread_model='elfouhaily',
               wind_dir=np.deg2rad(0.), wind_fetch=500.e3, wind_U=10.,
               current_mag=0., current_dir=np.deg2rad(0.),
               swell_enable=False, swell_ampl=4., swell_dir=np.deg2rad(0.), swell_wl=200.,
               compute=['Diff2'], opt_res=True, fft_max_prime=3, choppy_enable=True)
    
    # Obtain slopes at t=0s
    t = 0.0
    ocean.t = t

    plt.figure()
    plt.imshow(ocean.Diffxx, origin='lower', cmap=utils.sea_cmap)
    plt.colorbar()
    plt.title('XX - Second spatial derivative')
    plt.show()

    plt.figure()
    plt.imshow(ocean.Diffyy, origin='lower', cmap=utils.sea_cmap)
    plt.colorbar()
    plt.title('YY - Second spatial derivative')
    plt.show()
    
    plt.figure()
    plt.imshow(ocean.Diffxy, origin='lower', cmap=utils.sea_cmap)
    plt.colorbar()
    plt.title('XY - Second spatial derivative')
    plt.show()

if __name__ == '__main__':
    example()