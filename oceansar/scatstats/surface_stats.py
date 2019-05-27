__author__ = "Paco Lopez Dekker"
__email__ = "F.LopezDekker@tudeft.nl"


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import numexpr as ne
from oceansar import ocs_io
from oceansar import utils
import oceansar.utils.geometry as geosar
from oceansar import constants as const

from oceansar import nrcs as rcs
from oceansar.surfaces import OceanSurface
from oceansar import closure
import matplotlib as mpl

def surface_rel(cfg_file=None, inc_deg=None, ntimes=2, t_step=10e-3):
    """ This function generates a (short) time series of surface realizations.

        :param scf_file: the full path to the configuration with all OCEANSAR parameters
        :param inc_deg: the incident angle, in degree
        :param ntimes: number of time samples generated.
        :param t_step: spacing between time samples. This can be interpreted as the Pulse Repetition Interval

        :returns: a tuple with the configuration object, the surfaces, the radial velocities for each grid point,
                  and the complex scattering coefficients
    """

    cfg_file = utils.get_parFile(parfile=cfg_file)
    cfg = ocs_io.ConfigFile(cfg_file)
    use_hmtf = cfg.srg.use_hmtf
    scat_spec_enable = cfg.srg.scat_spec_enable
    scat_spec_mode = cfg.srg.scat_spec_mode
    scat_bragg_enable = cfg.srg.scat_bragg_enable
    scat_bragg_model = cfg.srg.scat_bragg_model
    scat_bragg_d = cfg.srg.scat_bragg_d
    scat_bragg_spec = cfg.srg.scat_bragg_spec
    scat_bragg_spread = cfg.srg.scat_bragg_spread

    # SAR
    inc_angle = np.deg2rad(cfg.sar.inc_angle)
    alt = cfg.sar.alt
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    pol = cfg.sar.pol
    l0 = const.c / f0
    k0 = 2.*np.pi*f0/const.c
    if pol == 'DP':
        do_hh = True
        do_vv = True
    elif pol == 'hh':
        do_hh = True
        do_vv = False
    else:
        do_hh = False
        do_vv = True
    # OCEAN / OTHERS
    ocean_dt = cfg.ocean.dt
    surface = OceanSurface()
    compute = ['D', 'Diff', 'Diff2','V','A']
    if use_hmtf:
        compute.append('hMTF')
    surface.init(cfg.ocean.Lx, cfg.ocean.Ly, cfg.ocean.dx,
                 cfg.ocean.dy, cfg.ocean.cutoff_wl,
                 cfg.ocean.spec_model, cfg.ocean.spread_model,
                 np.deg2rad(cfg.ocean.wind_dir),
                 cfg.ocean.wind_fetch, cfg.ocean.wind_U,
                 cfg.ocean.current_mag,
                 np.deg2rad(cfg.ocean.current_dir),
                 cfg.ocean.swell_enable, cfg.ocean.swell_ampl,
                 np.deg2rad(cfg.ocean.swell_dir),
                 cfg.ocean.swell_wl,
                 compute, cfg.ocean.opt_res,
                 cfg.ocean.fft_max_prime,
                 choppy_enable=cfg.ocean.choppy_enable)
    # Get a surface realization calculated
    surface.t = 0
    return surface


if __name__ == '__main__':
    cfg_file = utils.get_parFile(parfile=None)
    surface = surface_rel(cfg_file)