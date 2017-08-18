#!/usr/bin/env python

""" ===================================
    SAR ATI Processor (:mod:`ati_processor`)
    ===================================

    Perform ATI (*Along-Track Interferometry*) analysis on processed raw data.

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
from oceansar import io as tpio
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


def ati_process(cfg_file, proc_output_file, ocean_file, output_file):

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR ATI Processor: [%Y-%m-%d %H:%M:%S]", time.localtime()))
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
    # ATI
    rg_ml = cfg.ati.rg_ml
    az_ml = cfg.ati.az_ml
    ml_win = cfg.ati.ml_win
    plot_save = cfg.ati.plot_save
    plot_path = cfg.ati.plot_path
    plot_format = cfg.ati.plot_format
    plot_tex = cfg.ati.plot_tex
    plot_surface = cfg.ati.plot_surface
    plot_proc_ampl = cfg.ati.plot_proc_ampl
    plot_coh = cfg.ati.plot_coh
    plot_coh_all = cfg.ati.plot_coh_all
    plot_ati_phase = cfg.ati.plot_ati_phase
    plot_ati_phase_all = cfg.ati.plot_ati_phase_all
    plot_vel_hist = cfg.ati.plot_vel_hist
    plot_vel = cfg.ati.plot_vel

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

    # Some bookkeeping information
    output.write('--------------------------------------------\n')
    output.write('Variance of surface height = %.4f\n' % np.var(surface.Dz))
    output.write('Stdev of surface height = %.4f\n' % np.std(surface.Dz))
    output.write('--------------------------------------------\n\n')

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
    sr0 = geosar.inc_to_sr(inc_angle, alt)
    avg_az_shift = - v_radial_surf_mean / v_ground * sr0
    std_az_shift = v_radial_surf_std / v_ground * sr0
    ##################
    # ATI PROCESSING #
    ##################

    print('Starting ATI processing...')

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
    # .reshape((ch_dim) + (az_max - az_min, rg_max - rg_min))
    interfs = []
    cohs = []
    tind = 0
    coh_lut = np.zeros((proc_data.shape[0], proc_data.shape[0]), dtype=int)
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
            plt.imshow(utils.db(i_all[pind]), aspect='equal',
                       origin='lower',
                       vmin=utils.db(np.max(i_all[pind]))-20,
                       extent=[0., rg_span, 0., az_span], interpolation='nearest')
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
                       extent=[0., rg_span, 0., az_span], interpolation='nearest')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            plt.savefig(save_path)

    if plot_coh and ch_dim[0] > 1:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'ATI_coh_' +
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

    # ATI PHASE

    tau_ati = dist_chan/v_ground

    ati_phases = []
    # Hack to avoid interferogram computation if there are no interferometric channels
    if num_ch > 1:
        npol_ = npol
    else:
        npol_ = 0
    for pind in range(npol_):
        save_path = (plot_path + os.sep + 'ATI_pha_' +
                     polt[pind] + polt[pind] +
                     '.' + plot_format)
        coh_ind = coh_lut[(pind, pind + npol)]
        ati_phase = uwphase(cohs[coh_ind])
        ati_phases.append(ati_phase)
        v_radial_est = -ati_phase / tau_ati / (k0 * 2.)
        if plot_ati_phase:
            phase_mean = np.mean(ati_phase)
            phase_std = np.std(ati_phase)
            vmin = np.max([-np.abs(phase_mean) - 4*phase_std,
                           -np.abs(ati_phase).max()])
            vmax = np.min([np.abs(phase_mean) + 4*phase_std,
                           np.abs(ati_phase).max()])
            plt.figure()
            plt.imshow(ati_phase, aspect='equal',
                       origin='lower',
                       vmin=vmin, vmax=vmax,
                       extent=[0., rg_span, 0., az_span],
                       cmap='hsv')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("ATI Phase")
            plt.colorbar()
            plt.savefig(save_path)

            save_path = (plot_path + os.sep + 'ATI_rvel_' +
                         polt[pind] + polt[pind] +
                         '.' + plot_format)
            vmin = -np.abs(v_radial_surf_mean) - 4. * v_radial_surf_std
            vmax = np.abs(v_radial_surf_mean) + 4. * v_radial_surf_std
            plt.figure()
            plt.imshow(v_radial_est, aspect='equal',
                       origin='lower',
                       vmin=vmin, vmax=vmax,
                       extent=[0., rg_span, 0., az_span],
                       cmap='bwr')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Estimated Radial Velocity " + polt[pind])
            plt.colorbar()
            plt.savefig(save_path)

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
        ati_phase = uwphase(cohs[coh_ind])
        phase_mean = np.mean(ati_phase)
        phase_std = np.std(ati_phase)
        vmin = np.max([-np.abs(phase_mean) - 4*phase_std, -np.pi])
        vmax = np.min([np.abs(phase_mean) + 4*phase_std, np.pi])
        utils.image(ati_phase, aspect='equal',
                    min=vmin,  max=vmax,
                    cmap=utils.bwr_cmap, extent=[0., rg_span, 0., az_span],
                    xlabel='Ground range [m]', ylabel='Azimuth [m]',
                    title='XPOL Phase', cbar_xlabel='[rad]',
                    usetex=plot_tex, save=plot_save, save_path=save_path)

    if num_ch > 1:
        ati_phases = np.array(ati_phases)

        output.write('--------------------------------------------\n')
        output.write('SURFACE RADIAL VELOCITY - NO SMOOTHING\n')
        output.write('MEAN(SURF. V) = %.4f\n' % v_radial_surf_mean)
        output.write('STD(SURF. V) = %.4f\n' % v_radial_surf_std)
        output.write('--------------------------------------------\n\n')

        output.write('--------------------------------------------\n')
        output.write('SURFACE RADIAL VELOCITY - SMOOTHING (WIN. SIZE=%dx%d)\n' % (az_ml, rg_ml))
        output.write('MEAN(SURF. V) = %.4f\n' % v_radial_surf_mean)
        output.write('STD(SURF. V) = %.4f\n' % v_radial_surf_ml_std)
        output.write('--------------------------------------------\n\n')

        output.write('--------------------------------------------\n')
        output.write('SURFACE HORIZONTAL VELOCITY - NO SMOOTHING\n')
        output.write('MEAN(SURF. V) = %.4f\n' % v_horizo_surf_mean)
        output.write('STD(SURF. V) = %.4f\n' % v_horizo_surf_std)
        output.write('--------------------------------------------\n\n')

        if plot_vel_hist:
            # PLOT RADIAL VELOCITY
            plt.figure()

            plt.hist(v_radial_surf.flatten(), 200, normed=True, histtype='step')
            #plt.hist(v_radial_surf_ml.flatten(), 500, normed=True, histtype='step')
            plt.grid(True)
            plt.xlim([-np.abs(v_radial_surf_mean) - 4.*v_radial_surf_std, np.abs(v_radial_surf_mean) + 4.* v_radial_surf_std])
            plt.xlabel('Radial velocity [m/s]')
            plt.ylabel('PDF')
            plt.title('Surface velocity')

            if plot_save:
                plt.savefig(plot_path + os.sep + 'TRUE_radial_vel_hist.' + plot_format)
                plt.close()
            else:
                plt.show()

            plt.figure()
            plt.hist(v_radial_surf_ml.flatten(), 200, normed=True, histtype='step')
            #plt.hist(v_radial_surf_ml.flatten(), 500, normed=True, histtype='step')
            plt.grid(True)
            plt.xlim([-np.abs(v_radial_surf_mean) - 4.*v_radial_surf_std, np.abs(v_radial_surf_mean) + 4.* v_radial_surf_std])
            plt.xlabel('Radial velocity [m/s]')
            plt.ylabel('PDF')
            plt.title('Surface velocity (low pass filtered)')

            if plot_save:
                plt.savefig(plot_path + os.sep + 'TRUE_radial_vel_ml_hist.' + plot_format)
                plt.close()
            else:
                plt.show()

        if plot_vel:

            utils.image(v_radial_surf, aspect='equal', cmap=utils.bwr_cmap, extent=[0., rg_span, 0., az_span],
                        xlabel='Ground range [m]', ylabel='Azimuth [m]', title='Surface Radial Velocity', cbar_xlabel='[m/s]',
                        min=-np.abs(v_radial_surf_mean) - 4.*v_radial_surf_std, max=np.abs(v_radial_surf_mean) + 4.*v_radial_surf_std,
                        usetex=plot_tex, save=plot_save, save_path=plot_path + os.sep + 'TRUE_radial_vel.' + plot_format)
            utils.image(v_radial_surf_ml, aspect='equal', cmap=utils.bwr_cmap, extent=[0., rg_span, 0., az_span],
                        xlabel='Ground range [m]', ylabel='Azimuth [m]', title='Surface Radial Velocity', cbar_xlabel='[m/s]',
                        min=-np.abs(v_radial_surf_mean) - 4.*v_radial_surf_std, max=np.abs(v_radial_surf_mean) + 4.*v_radial_surf_std,
                        usetex=plot_tex, save=plot_save, save_path=plot_path + os.sep + 'TRUE_radial_vel_ml.' + plot_format)

        ##  ESTIMATED VELOCITIES

        # Note: plot limits are taken from surface calculations to keep the same ranges

        # ESTIMATE RADIAL VELOCITY
        v_radial_ests = -ati_phases/tau_ati/(k0*2.)

        # ESTIMATE HORIZONTAL VELOCITY
        v_horizo_ests = -ati_phases/tau_ati/(k0*2.)/np.sin(inc_angle)

        #Trim edges
        v_radial_ests = v_radial_ests[:, az_guard:-az_guard, 5:-5]
        v_horizo_ests = v_horizo_ests[:, az_guard:-az_guard, 5:-5]
        output.write('--------------------------------------------\n')
        output.write('ESTIMATED RADIAL VELOCITY - NO SMOOTHING\n')
        for pind in range(npol):
            output.write("%s Polarization\n" % polt[pind])
            output.write('MEAN(EST. V) = %.4f\n' % np.mean(v_radial_ests[pind]))
            output.write('STD(EST. V) = %.4f\n' % np.std(v_radial_ests[pind]))
        output.write('--------------------------------------------\n\n')

        output.write('--------------------------------------------\n')
        output.write('ESTIMATED RADIAL VELOCITY - SMOOTHING (WIN. SIZE=%dx%d)\n' % (az_ml, rg_ml))
        for pind in range(npol):
            output.write("%s Polarization\n" % polt[pind])
            output.write('MEAN(EST. V) = %.4f\n' % np.mean(utils.smooth(utils.smooth(v_radial_ests[pind],
                                                                                     rg_ml, axis=1),
                                                                        az_ml, axis=0)))
            output.write('STD(EST. V) = %.4f\n' % np.std(utils.smooth(utils.smooth(v_radial_ests[pind],
                                                                                   rg_ml, axis=1),
                                                                      az_ml, axis=0)))
        output.write('--------------------------------------------\n\n')

        output.write('--------------------------------------------\n')
        output.write('ESTIMATED HORIZONTAL VELOCITY - NO SMOOTHING\n')
        for pind in range(npol):
            output.write("%s Polarization\n" % polt[pind])
            output.write('MEAN(EST. V) = %.4f\n' % np.mean(v_horizo_ests[pind]))
            output.write('STD(EST. V) = %.4f\n' % np.std(v_horizo_ests[pind]))
        output.write('--------------------------------------------\n\n')

    # Processed NRCS

    NRCS_est_avg = 10*np.log10(np.mean(np.mean(i_all[:, az_guard:-az_guard, 5:-5], axis=-1), axis=-1))
    output.write('--------------------------------------------\n')
    for pind in range(npol):
        output.write("%s Polarization\n" % polt[pind])
        output.write('Estimated mean NRCS = %5.2f\n' % NRCS_est_avg[pind])
    output.write('--------------------------------------------\n\n')

    # Some bookkeeping information
    output.write('--------------------------------------------\n')
    output.write('GROUND RANGE GRID SPACING = %.4f\n' % grg_grid_spacing)
    output.write('AZIMUTH GRID SPACING = %.4f\n' % az_grid_spacing)
    output.write('--------------------------------------------\n\n')

    output.close()

    if plot_vel_hist and num_ch > 1:
        # PLOT RADIAL VELOCITY
        plt.figure()
        plt.hist(v_radial_surf.flatten(), 200, normed=True, histtype='step',
                 label='True')
        for pind in range(npol):
            plt.hist(v_radial_ests[pind].flatten(), 200, normed=True,
                     histtype='step', label=polt[pind])
        plt.grid(True)
        plt.xlim([-np.abs(v_radial_surf_mean) - 4.*v_radial_surf_std,
                  np.abs(v_radial_surf_mean) + 4.*v_radial_surf_std])
        plt.xlabel('Radial velocity [m/s]')
        plt.ylabel('PDF')
        plt.title('Estimated velocity')
        plt.legend()

        if plot_save:
            plt.savefig(plot_path + os.sep + 'ATI_radial_vel_hist.' + plot_format)
            plt.close()
        else:
            plt.show()

    print('----------------------------------------')
    print(time.strftime("ATI Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('----------------------------------------')


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-p', '--proc_file')
    parser.add_argument('-s', '--ocean_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    ati_process(args.cfg_file, args.proc_file, args.ocean_file, args.output_file)
