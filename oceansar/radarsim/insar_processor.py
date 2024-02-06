#!/usr/bin/env python

""" ===================================
    InSAR Processor (:mod:`ati_processor`)
    ===================================

    InSAR processing (flattening, ml, etc) of SLC data.

    **Arguments**
        * -c, --cfg_file: Configuration file
        * -p, --proc_file: Processed raw data file
        * -s, --ocean_file: Ocean state file
        * -o, --output_file: Output file

"""

import os
import time
import argparse

import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

from oceansar.utils import geometry as geosar
from oceansar import utils
from oceansar.utils.geometry import sr_to_geo
from oceansar import ocs_io as tpio
from oceansar import constants as const
from oceansar.surfaces import OceanSurface


def uwphase(phasor):
    """ This calculates the phase of an array of phasors
    """
    pha = np.angle(phasor)
    nbins = np.min([72, np.ceil(phasor.size / 50).astype(int)])
    pha_hist, h_edgs = np.histogram(pha, bins=nbins, range=(-np.pi, np.pi))
    pha_mod = (h_edgs[pha_hist.argmax()] + h_edgs[pha_hist.argmax()]) / 2
    pha = np.angle(phasor * np.exp(-1j * pha_mod)) + pha_mod
    return pha


def insar_process(cfg_file, proc_output_file, ocean_file, output_file):

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR InSAR Processor: [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-------------------------------------------------------------------')

    print('Initializing...')

    ## CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    pol = cfg.sar.pol
    if pol == 'DP':
        polt = ['hh', 'vv']
    elif pol == 'hh':
        polt = ['hh']
    else:
        polt = ['vv']
    # ATI
    rg_ml = cfg.insar.rg_ml
    az_ml = cfg.insar.az_ml
    ml_win = cfg.insar.ml_win
    flatten = cfg.insar.flatten
    plot_save = cfg.insar.plot_save
    plot_path = cfg.insar.plot_path
    plot_format = cfg.insar.plot_format
    plot_tex = cfg.insar.plot_tex
    plot_surface = cfg.insar.plot_surface
    plot_proc_ampl = cfg.insar.plot_proc_ampl
    plot_coh = cfg.insar.plot_coh
    plot_coh_all = cfg.insar.plot_coh_all

    # PROCESSED DATA
    proc_content = tpio.ProcFile(proc_output_file, 'r')
    proc_data = proc_content.get('slc*')
    sr0 = proc_content.get('sr0')
    inc_angle = proc_content.get('inc_angle')
    b_ati = proc_content.get('b_ati')
    b_xti = proc_content.get('b_xti')
    f0 = proc_content.get('f0')
    prf = proc_content.get('prf')
    num_ch = proc_content.get('num_ch')
    rg_bw = proc_content.get('rg_bw')
    rg_sampling = proc_content.get('rg_sampling')
    v_ground = proc_content.get('v_ground')
    alt = proc_content.get('orbit_alt')
    inc_angle = np.deg2rad(proc_content.get('inc_angle'))
    proc_content.close()

    ## CALCULATE PARAMETERS
    if v_ground == 'auto': v_ground = geosar.orbit_to_vel(alt, ground=True)
    k0 = 2.*np.pi*f0/const.c


    # OCEAN SURFACE
    surface = OceanSurface()
    surface.load(ocean_file, compute=['D', 'V'])
    surface.t = 0.

    # OUTPUT FILE
    # output = open(output_file, 'w')

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

    # SURFACE HORIZONTAL VELOCITY
    v_horizo_surf = surface.Vx
    v_horizo_surf_ml = utils.smooth(utils.smooth(v_horizo_surf, res_fact * rg_ml, axis=1), res_fact * az_ml, axis=0)
    v_horizo_surf_mean = np.mean(v_horizo_surf)
    v_horizo_surf_std = np.std(v_horizo_surf)
    v_horizo_surf_ml_std = np.std(v_horizo_surf_ml)

    # Expected mean azimuth shift
    # sr0 = geosar.inc_to_sr(inc_angle, alt)
    avg_az_shift = - v_radial_surf_mean / v_ground * sr0
    std_az_shift = v_radial_surf_std / v_ground * sr0
    ##################
    # InSAR PROCESSING #
    ##################

    print('Starting InSAR processing...')

    # Get dimensions & calculate region of interest
    rg_span = surface.Lx
    az_span = surface.Ly
    rg_size = proc_data[0].shape[2]
    az_size = proc_data[0].shape[1]

    # Note: RG is projected, so plots are Ground Range
    rg_min = 0
    rg_max = int(rg_span/(const.c/2./rg_sampling/np.sin(inc_angle)))
    az_min = int(az_size/2. + (-az_span/2. + avg_az_shift)/(v_ground/prf))
    az_max = int(az_size/2. + (az_span/2. + avg_az_shift)/(v_ground/prf))
    az_guard = int(std_az_shift / (v_ground / prf))
    if (az_max - az_min) < (2 * az_guard - 10):
        print('Not enough edge-effect free image')
        return

    inter_chan_shift_dist = b_ati / (v_ground/prf)
    # Subsample shift in azimuth
    for chind in range(proc_data.shape[0]):
        shift_dist = - inter_chan_shift_dist[chind]
        shift_arr = np.exp(-2j * np.pi * shift_dist *
                           np.roll(np.arange(az_size) - az_size/2,
                                   int(-az_size / 2)) / az_size)
        shift_arr = shift_arr.reshape((1, az_size, 1))
        proc_data[chind] = np.fft.ifft(np.fft.fft(proc_data[chind], axis=1) *
                                       shift_arr, axis=1)
    # Flat earth
    if flatten:
        slant_range = sr0 + np.arange(rg_size) / rg_sampling * const.c/2
        gr_interp, inc, theta_l_interp, b_interp = sr_to_geo(slant_range, alt)
        flatearth_dinc = (inc - inc_angle)
        # print(flatearth_dinc[0:4])
        # Add cross-track phase
        flattening_phasor = np.zeros((num_ch, 1, 1, rg_size), dtype=complex)
        for ch in range(num_ch):
            flattening_phasor[ch, 0, 0, :] = np.exp(1j * k0 * b_xti[ch] * flatearth_dinc)
        proc_data = proc_data * flattening_phasor
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
    # .reshape((ch_dim) + (az_max - az_min, rg_max - rg_min))
    interfs = []
    cohs = []
    tind = 0
    coh_lut = np.zeros((proc_data.shape[0], proc_data.shape[0]), dtype=int) -1
    for chind1 in range(proc_data.shape[0]):
        for chind2 in range(chind1 + 1, proc_data.shape[0]):
            coh_lut[chind1, chind2] = tind
            tind = tind + 1
            t_interf = utils.smooth(utils.smooth(proc_data[chind2] *
                                                 np.conj(proc_data[chind1]),
                                                 rg_ml, axis=1, window=ml_win),
                                    az_ml, axis=0, window=ml_win)
            interfs.append(t_interf[az_min:az_max, rg_min:rg_max])
            cohs.append(t_interf[az_min:az_max, rg_min:rg_max] /
                        np.sqrt(i_all[chind1] * i_all[chind2]))
    # coh_lut = coh_lut.reshape((num_ch, npol, num_ch, npol))
    i_all = i_all.reshape(ch_dim + (az_max - az_min, rg_max - rg_min))
    cohs = np.array(cohs)
    l1b_file = tpio.L1bFile(output_file, 'w', i_all.shape)
    l1b_file.set('ml_intensity', i_all)
    l1b_file.set('ml_coherence', np.abs(cohs))
    l1b_file.set('ml_phase', np.angle(cohs))
    l1b_file.set('coh_lut', coh_lut.reshape((num_ch, npol, num_ch, npol)))
    l1b_file.set('inc_angle', inc_angle)
    l1b_file.set('f0', f0)
    l1b_file.set('num_ch', num_ch)
    l1b_file.set('az_sampling', prf)
    l1b_file.set('v_ground', v_ground)
    l1b_file.set('orbit_alt', alt)
    l1b_file.set('sr0', sr0)
    l1b_file.set('rg_sampling', rg_sampling)
    # l1b_file.set('rg_bw', rg_bw)
    l1b_file.set('b_ati', b_ati)
    l1b_file.set('b_xti', b_xti)
    l1b_file.set('rg_ml', rg_ml)
    l1b_file.set('az_ml', az_ml)
    l1b_file.close()

    print('Generating plots and estimating values...')

    # SURFACE HEIGHT
    if plot_surface:
        plt.figure()
        plt.imshow(surface.Dz, cmap="ocean",
                   extent=[0, surface.Lx, 0, surface.Ly], origin='lower')
        plt.title('Surface Height')
        plt.xlabel('Ground range [m]')
        plt.ylabel('Azimuth [m]')
        cbar = plt.colorbar()
        cbar.ax.set_xlabel('[m]')

        if plot_save:
            plt.savefig(plot_path + os.sep + 'plot_surface.' + plot_format,
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    # PROCESSED AMPLITUDE
    if plot_proc_ampl:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'amp_dB_' + polt[pind]+
                         '.' + plot_format)
            plt.figure()
            plt.imshow(utils.db(i_all[0, pind]), aspect='equal',
                       origin='lower',
                       vmin=utils.db(np.max(i_all[pind]))-20,
                       extent=[0., rg_span, 0., az_span], interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            plt.savefig(save_path)
            plt.close()

            save_path = (plot_path + os.sep + 'amp_' + polt[pind]+
                         '.' + plot_format)
            int_img = (i_all[0, pind])**0.5
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
            plt.close()

    if plot_coh and ch_dim[0] > 1:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'coh_' +
                         polt[pind] + polt[pind] +
                         '.' + plot_format)
            coh_ind = coh_lut[(pind, pind + npol)]
            plt.figure()
            plt.imshow(np.abs(cohs[coh_ind]), aspect='equal',
                       origin='lower',
                       vmin=0, vmax=1,
                       extent=[0., rg_span, 0., az_span],
                       cmap='bone')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("ATI Coherence")
            # plt.colorbar()
            plt.savefig(save_path)
            plt.close()

    if num_ch > 1:
        npol_ = npol
    else:
        npol_ = 0

    insar_phases = []
    for pind in range(npol_):
        save_path = (plot_path + os.sep + 'pha_' +
                     polt[pind] + polt[pind] +
                     '.' + plot_format)
        coh_ind = coh_lut[(pind, pind + npol)]
        insar_phase = uwphase(cohs[coh_ind])
        insar_phases.append(insar_phase)
        # v_radial_est = -ati_phase / tau_ati[1] / (k0 * 2.)

        phase_mean = np.mean(insar_phase)
        phase_std = np.std(insar_phase)
        vmin = np.max([-np.abs(phase_mean) - 4*phase_std,
                       -np.abs(insar_phase).max()])
        vmax = np.min([np.abs(phase_mean) + 4*phase_std,
                       np.abs(insar_phase).max()])
        plt.figure()
        plt.imshow(insar_phase, aspect='equal',
                   origin='lower',
                   vmin=vmin, vmax=vmax,
                   extent=[0., rg_span, 0., az_span],
                   cmap='hsv')
        plt.xlabel('Ground range [m]')
        plt.ylabel('Azimuth [m]')
        plt.title("InSAR Phase")
        plt.colorbar()
        plt.savefig(save_path)
        plt.close()



    if npol_ == 4:  # Bypass this for now
        # Cross pol interferogram
        coh_ind = coh_lut[(0, 1)]
        save_path = (plot_path + os.sep + 'POL_coh_' +
                     polt[0] + polt[1] +
                     '.' + plot_format)
        utils.image(np.abs(cohs[coh_ind]), max=1, min=0, aspect='equal',
                    cmap='gray', extent=[0., rg_span, 0., az_span],
                    xlabel='Ground range [m]', ylabel='Azimuth [m]',
                    title='XPOL Coherence',
                    usetex=plot_tex, save=plot_save, save_path=save_path)
        save_path = (plot_path + os.sep + 'POL_pha_' +
                     polt[0] + polt[1] +
                     '.' + plot_format)
        insar_phase = uwphase(cohs[coh_ind])
        phase_mean = np.mean(insar_phase)
        phase_std = np.std(insar_phase)
        vmin = np.max([-np.abs(phase_mean) - 4*phase_std, -np.pi])
        vmax = np.min([np.abs(phase_mean) + 4*phase_std, np.pi])
        utils.image(insar_phase, aspect='equal',
                    min=vmin,  max=vmax,
                    cmap=utils.bwr_cmap, extent=[0., rg_span, 0., az_span],
                    xlabel='Ground range [m]', ylabel='Azimuth [m]',
                    title='XPOL Phase', cbar_xlabel='[rad]',
                    usetex=plot_tex, save=plot_save, save_path=save_path)

    print('----------------------------------------')
    print(time.strftime("InSAR L1b Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('----------------------------------------')


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-p', '--proc_file')
    parser.add_argument('-s', '--ocean_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    insar_process(args.cfg_file, args.proc_file, args.ocean_file, args.output_file)
