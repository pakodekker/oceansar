""" OSIRIS Test """

import numpy as np
import matplotlib.pyplot as plt
from osiris.surfaces import OceanSurface

from trampa import utils
        
def example():
    
    # Create ocean surface   
    ocean = OceanSurface()
    ocean.init(Lx=512., Ly=512., dx=1., dy=1., cutoff_wl='auto',
               spec_model='elfouhaily', spread_model='elfouhaily',
               wind_dir=np.deg2rad(0.), wind_fetch=500.e3, wind_U=8.,
               current_mag=0., current_dir=np.deg2rad(0.),
               swell_enable=True, swell_ampl=4., swell_dir=np.deg2rad(0.), swell_wl=200.,
               compute=['D'], opt_res=True, fft_max_prime=3)
    
    # Plot height field for t=[0,10)s
    #plt.ion()
    plt.figure()
    for t in np.arange(0., 10., 0.1):
        ocean.t = t
        
        plt.clf()
        plt.imshow(ocean.Dz, origin='lower', cmap=utils.sea_cmap)
        plt.colorbar()
        plt.title('Height field, t=%.4f' % t)

        plt.show()
        #plt.draw()
        
if __name__ == '__main__':
    example()