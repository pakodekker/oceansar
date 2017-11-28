#!/usr/bin/env python

""" ===================================
    SAR ATI Processor (:mod:`L2_wavespectrum`)
    ===================================

    Compute wavespectra from data.

    **Arguments**
        * -c, --cfg_file: Configuration file
        * -p, --proc_file: Processed raw data file
        * -s, --ocean_file: Ocean state file
        * -o, --output_file: Output file

"""

import os
import time
import argparse
import pickle

import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

from oceansar.utils import geometry as geosar
from oceansar import utils
from oceansar import io as tpio
from oceansar import constants as const
from oceansar.surfaces import OceanSurface

__author__ = "Paco Lopez Dekker"
__email__ = "F.LopezDekker@tudeft.nl"


def l2_wavespectrum(cfg_file, proc_output_file, ocean_file, output_file):

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR L2 Wavespectra: [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-------------------------------------------------------------------')

    print('Initializing...')

    ## CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    # SAR
    inc_angle = np.deg2rad(cfg.sar.inc_angle)
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    num_ch = cfg.sar.num_ch
    ant_L = cfg.sar.ant_L
    alt = cfg.sar.alt
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.sar.rg_bw
    over_fs = cfg.sar.over_fs
    pol = cfg.sar.pol
    if pol == 'DP':
        polt = ['hh', 'vv']
    elif pol == 'hh':
        polt = ['hh']
    else:
        polt = ['vv']
        # L2 wavespectrum
    rg_ml = cfg.L2_wavespectrum.rg_ml
    az_ml = cfg.L2_wavespectrum.az_ml
    krg_ml = cfg.L2_wavespectrum.krg_ml
    kaz_ml = cfg.L2_wavespectrum.kaz_ml
    ml_win = cfg.L2_wavespectrum.ml_win
    plot_save = cfg.L2_wavespectrum.plot_save
    plot_path = cfg.L2_wavespectrum.plot_path
    plot_format = cfg.L2_wavespectrum.plot_format
    plot_tex = cfg.L2_wavespectrum.plot_tex
    plot_surface = cfg.L2_wavespectrum.plot_surface
    plot_proc_ampl = cfg.L2_wavespectrum.plot_proc_ampl
    plot_spectrum = cfg.L2_wavespectrum.plot_spectrum
    n_sublook = cfg.L2_wavespectrum.n_sublook
    sublook_weighting = cfg.L2_wavespectrum.sublook_az_weighting

    ## CALCULATE PARAMETERS
    if v_ground == 'auto': v_ground = geosar.orbit_to_vel(alt, ground=True)
    k0 = 2.*np.pi*f0/const.c
    rg_sampling = rg_bw*over_fs

    # PROCESSED RAW DATA
    proc_content = tpio.ProcFile(proc_output_file, 'r')
    proc_data = proc_content.get('slc*')
    proc_content.close()

    # OCEAN SURFACE
    surface = OceanSurface()
    surface.load(ocean_file, compute=['D', 'V'])
    surface.t = 0.

    # OUTPUT FILE
    output = open(output_file, 'w')

    # OTHER INITIALIZATIONS
    # Enable TeX
    if plot_tex:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)

    # Create plots directory
    plot_path = os.path.dirname(output_file) + os.sep + plot_path
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    # SURFACE VELOCITIES
    grg_grid_spacing = (const.c/2./rg_sampling/np.sin(inc_angle))
    rg_res_fact = grg_grid_spacing / surface.dx
    az_grid_spacing = (v_ground/prf)
    az_res_fact = az_grid_spacing / surface.dy
    res_fact = np.ceil(np.sqrt(rg_res_fact*az_res_fact))

    # SURFACE RADIAL VELOCITY
    v_radial_surf = surface.Vx*np.sin(inc_angle) - surface.Vz*np.cos(inc_angle)
    v_radial_surf_ml = utils.smooth(utils.smooth(v_radial_surf, res_fact * rg_ml, axis=1), res_fact * az_ml, axis=0)
    v_radial_surf_mean = np.mean(v_radial_surf)
    v_radial_surf_std = np.std(v_radial_surf)
    v_radial_surf_ml_std = np.std(v_radial_surf_ml)

    # Expected mean azimuth shift
    sr0 = geosar.inc_to_sr(inc_angle, alt)
    avg_az_shift = - v_radial_surf_mean / v_ground * sr0
    std_az_shift = v_radial_surf_std / v_ground * sr0


    print('Starting Wavespectrum processing...')

    # Get dimensions & calculate region of interest
    rg_span = surface.Lx
    az_span = surface.Ly
    rg_size = proc_data[0].shape[2]
    az_size = proc_data[0].shape[1]

    # Note: RG is projected, so plots are Ground Range
    rg_min = 0
    rg_max = np.int(rg_span/(const.c/2./rg_sampling/np.sin(inc_angle)))
    az_min = np.int(az_size/2. + (-az_span/2. + avg_az_shift)/(v_ground/prf))
    az_max = np.int(az_size/2. + (az_span/2. + avg_az_shift)/(v_ground/prf))
    az_guard = np.int(std_az_shift / (v_ground / prf))
    az_min = az_min + az_guard
    az_max = az_max - az_guard
    if (az_max - az_min) < (2 * az_guard - 10):
        print('Not enough edge-effect free image')
        return

    # Adaptive coregistration
    if cfg.sar.L_total:
        ant_L = ant_L/np.float(num_ch)
        dist_chan = ant_L/2
    else:
        if np.float(cfg.sar.Spacing) != 0:
            dist_chan = np.float(cfg.sar.Spacing)/2
        else:
            dist_chan = ant_L/2
    # dist_chan = ant_L/num_ch/2.
    print('ATI Spacing: %f' % dist_chan)
    inter_chan_shift_dist = dist_chan/(v_ground/prf)
    # Subsample shift in azimuth
    for chind in range(proc_data.shape[0]):
        shift_dist = - chind * inter_chan_shift_dist
        shift_arr = np.exp(-2j * np.pi * shift_dist *
                           np.roll(np.arange(az_size) - az_size/2,
                                   int(-az_size / 2)) / az_size)
        shift_arr = shift_arr.reshape((1, az_size, 1))
        proc_data[chind] = np.fft.ifft(np.fft.fft(proc_data[chind], axis=1) *
                                       shift_arr, axis=1)

    # First dimension is number of channels, second is number of pols
    ch_dim = proc_data.shape[0:2]
    npol = ch_dim[1]
    proc_data_rshp = [np.prod(ch_dim), proc_data.shape[2], proc_data.shape[3]]
    # Compute extended covariance matrices...
    proc_data = proc_data.reshape(proc_data_rshp)
    # Intensities
    i_all = []
    for chind in range(proc_data.shape[0]):
        this_i = utils.smooth(utils.smooth(np.abs(proc_data[chind])**2., rg_ml, axis=1, window=ml_win),
                              az_ml, axis=0, window=ml_win)
        i_all.append(this_i[az_min:az_max, rg_min:rg_max])
    i_all = np.array(i_all)

    ## Wave spectra computation
    ## Processed Doppler bandwidth
    proc_bw = cfg.processing.doppler_bw
    PRF = cfg.sar.prf
    fa = np.fft.fftfreq(proc_data_rshp[1], 1/PRF)
    # Filters
    sublook_filt = []
    sublook_bw = proc_bw / n_sublook
    for i_sbl in range(n_sublook):
        fa_min = -1 * proc_bw / 2 + i_sbl * sublook_bw
        fa_max = fa_min + sublook_bw
        fa_c = (fa_max + fa_min)/2
        win = np.where(np.logical_and(fa > fa_min, fa < fa_max),
                       (sublook_weighting - (1 - sublook_weighting) * np.cos(2 * np.pi * (fa - fa_min) / sublook_bw)),
                       0)
        sublook_filt.append(win)

    # Apply sublooks
    az_downsmp = int(np.floor(az_ml / 2))
    rg_downsmp = int(np.floor(rg_ml / 2))
    sublooks = []
    sublooks_f = []
    for i_sbl in range(n_sublook):
        # Go to frequency domain
        sublook_data = np.fft.ifft(np.fft.fft(proc_data, axis=1) *
                                   sublook_filt[i_sbl].reshape((1, proc_data_rshp[1], 1)), axis=1)
        # Get intensities
        sublook_data = np.abs(sublook_data)**2
        # Multilook
        for chind in range(proc_data.shape[0]):
            sublook_data[chind] = utils.smooth(utils.smooth(sublook_data[chind], rg_ml, axis=1), az_ml, axis=0)
        # Keep only valid part and down sample
        sublook_data = sublook_data[:, az_min:az_max:az_downsmp, rg_min:rg_max:rg_downsmp]
        sublooks.append(sublook_data)
        sublooks_f.append(np.fft.fft(np.fft.fft(sublook_data - np.mean(sublook_data), axis=1), axis=2))

    kaz = 2 * np.pi * np.fft.fftfreq(sublook_data.shape[1], az_downsmp * az_grid_spacing)
    kgrg = 2 * np.pi * np.fft.fftfreq(sublook_data.shape[2], rg_downsmp * grg_grid_spacing)

    xspecs = []
    tind = 0
    xspec_lut = np.zeros((len(sublooks), len(sublooks)), dtype=int)

    for ind1 in range(len(sublooks)):
        for ind2 in range(ind1 + 1, len(sublooks)):
            xspec_lut[ind1, ind2] = tind
            tind = tind + 1
            xspec = sublooks_f[ind1] * np.conj(sublooks_f[ind2])
            xspecs.append(xspec)


    with open(output_file, 'wb') as output:
        pickle.dump(xspecs, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump([kaz, kgrg], output, pickle.HIGHEST_PROTOCOL)

    # PROCESSED AMPLITUDE
    if plot_proc_ampl:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'amp_dB_' + polt[pind]+
                         '.' + plot_format)
            plt.figure()
            plt.imshow(utils.db(i_all[pind]), aspect='equal',
                       origin='lower',
                       vmin=utils.db(np.max(i_all[pind]))-20,
                       extent=[0., rg_span, 0., az_span], interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            plt.savefig(save_path)

            save_path = (plot_path + os.sep + 'amp_' + polt[pind]+
                         '.' + plot_format)
            int_img = (i_all[pind])**0.5
            vmin = np.mean(int_img) - 3 * np.std(int_img)
            vmax = np.mean(int_img) + 3 * np.std(int_img)
            plt.figure()
            plt.imshow(int_img, aspect='equal',
                       origin='lower',
                       vmin=vmin, vmax=vmax,
                       extent=[0., rg_span, 0., az_span], interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            plt.savefig(save_path)

    ## FIXME: I am plotting the cross spectrum for the first polarization and the first channel only, which is not
    ## very nice. To be fixed, in particular por multiple polarizations

    for ind1 in range(len(sublooks)):
        for ind2 in range(ind1 + 1, len(sublooks)):
            save_path_abs = os.path.join(plot_path, ('xspec_abs_%i%i.%s' % (ind1+1, ind2+1, plot_format)))
            save_path_pha = os.path.join(plot_path, ('xspec_pha_%i%i.%s' % (ind1+1, ind2+1, plot_format)))
            save_path_im = os.path.join(plot_path, ('xspec_im_%i%i.%s' % (ind1+1, ind2+1, plot_format)))
            ml_xspec = utils.smooth(utils.smooth(np.fft.fftshift(xspecs[xspec_lut[ind1, ind2]][0]), krg_ml, axis=1),
                                    kaz_ml, axis=0)
            plt.figure()
            plt.imshow(np.abs(ml_xspec), origin='lower', cmap='inferno_r',
                       extent=[kgrg.min(), kgrg.max(), kaz.min(), kaz.max()],
                       interpolation='nearest')
            plt.grid(True)
            pltax = plt.gca()
            pltax.set_xlim((-0.1, 0.1))
            pltax.set_ylim((-0.1, 0.1))
            northarr_length = 0.075  # np.min([surface_full.kx.max(), surface_full.ky.max()])
            pltax.arrow(0, 0,
                        -northarr_length * np.sin(np.radians(cfg.sar.heading)),
                        northarr_length * np.cos(np.radians(cfg.sar.heading)),
                        fc="k", ec="k")
            plt.xlabel('$k_x$ [rad/m]')
            plt.ylabel('$k_y$ [rad/m]')
            plt.colorbar()
            plt.savefig(save_path_abs)
            plt.close()
            plt.figure()
            ml_xspec_pha = np.angle(ml_xspec)
            ml_xspec_im = np.imag(ml_xspec)
            immax = np.abs(ml_xspec_im).max()
            whimmax = np.abs(ml_xspec_im).flatten().argmax()
            phmax = np.abs(ml_xspec_pha.flatten()[whimmax])
            plt.imshow(ml_xspec_pha, origin='lower', cmap='bwr',
                       extent=[kgrg.min(), kgrg.max(), kaz.min(), kaz.max()],
                       interpolation='nearest', vmin=-2*phmax, vmax=2*phmax)
            plt.grid(True)
            pltax = plt.gca()
            pltax.set_xlim((-0.1, 0.1))
            pltax.set_ylim((-0.1, 0.1))
            northarr_length = 0.075  # np.min([surface_full.kx.max(), surface_full.ky.max()])
            pltax.arrow(0, 0,
                        -northarr_length * np.sin(np.radians(cfg.sar.heading)),
                        northarr_length * np.cos(np.radians(cfg.sar.heading)),
                        fc="k", ec="k")
            plt.xlabel('$k_x$ [rad/m]')
            plt.ylabel('$k_y$ [rad/m]')
            plt.colorbar()
            plt.savefig(save_path_pha)
            plt.close()
            plt.figure()
            plt.imshow(ml_xspec_im, origin='lower', cmap='bwr',
                       extent=[kgrg.min(), kgrg.max(), kaz.min(), kaz.max()],
                       interpolation='nearest', vmin=-2 * immax, vmax=2 * immax)
            plt.grid(True)
            pltax = plt.gca()
            pltax.set_xlim((-0.1, 0.1))
            pltax.set_ylim((-0.1, 0.1))
            northarr_length = 0.075  # np.min([surface_full.kx.max(), surface_full.ky.max()])
            pltax.arrow(0, 0,
                        -northarr_length * np.sin(np.radians(cfg.sar.heading)),
                        northarr_length * np.cos(np.radians(cfg.sar.heading)),
                        fc="k", ec="k")
            plt.xlabel('$k_x$ [rad/m]')
            plt.ylabel('$k_y$ [rad/m]')
            plt.colorbar()
            plt.savefig(save_path_im)
            plt.close()




if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-p', '--proc_file')
    parser.add_argument('-s', '--ocean_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    l2_wavespectrum(args.cfg_file, args.proc_file, args.ocean_file, args.output_file)
