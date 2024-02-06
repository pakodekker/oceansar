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


def ati_process(cfg_file, insar_output_file, ocean_file, output_file):

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR ATI Processor: [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-------------------------------------------------------------------')

    print('Initializing...')

    ## CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    # SAR
    pol = cfg.sar.pol
    if pol == 'DP':
        polt = ['hh', 'vv']
    elif pol == 'hh':
        polt = ['hh']
    else:
        polt = ['vv']
    # ATI

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

    # PROCESSED InSAR L1b DATA
    insar_data = tpio.L1bFile(insar_output_file, 'r')
    i_all = insar_data.get('ml_intensity')
    cohs = insar_data.get('ml_coherence') * np.exp(1j * insar_data.get('ml_phase'))
    coh_lut = insar_data.get('coh_lut')
    sr0 = insar_data.get('sr0')
    inc_angle = insar_data.get('inc_angle')
    b_ati = insar_data.get('b_ati')
    b_xti = insar_data.get('b_xti')
    f0 = insar_data.get('f0')
    az_sampling = insar_data.get('az_sampling')
    num_ch = insar_data.get('num_ch')
    rg_sampling = insar_data.get('rg_sampling')
    v_ground = insar_data.get('v_ground')
    alt = insar_data.get('orbit_alt')
    inc_angle = np.deg2rad(insar_data.get('inc_angle'))
    rg_ml = insar_data.get('rg_ml')
    az_ml = insar_data.get('az_ml')
    insar_data.close()

    # CALCULATE PARAMETERS
    k0 = 2.*np.pi*f0/const.c

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
    az_grid_spacing = (v_ground/az_sampling)
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

    az_guard = int(std_az_shift / (v_ground / az_sampling))
    ##################
    # ATI PROCESSING #
    ##################

    print('Starting ATI processing...')

    # Get dimensions & calculate region of interest
    rg_span = surface.Lx
    az_span = surface.Ly

    # First dimension is number of channels, second is number of pols
    ch_dim = i_all.shape[0:2]
    npol = ch_dim[1]

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
            save_path = (plot_path + os.sep + 'amp_' + polt[pind]
                         + '.' + plot_format)
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
            save_path = (plot_path + os.sep + 'ATI_coh_'
                         + polt[pind] + polt[pind]
                         + '.' + plot_format)
            coh_ind = coh_lut[0, pind, 1, pind]
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

    tau_ati = b_ati/v_ground

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
        coh_ind = coh_lut[(0, pind, 1, pind)]
        ati_phase = uwphase(cohs[coh_ind])
        ati_phases.append(ati_phase)
        v_radial_est = -ati_phase / tau_ati[1] / (k0 * 2.)
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

            plt.hist(v_radial_surf.flatten(), 200, density=True, histtype='step')
            #plt.hist(v_radial_surf_ml.flatten(), 500, density=True, histtype='step')
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
            plt.hist(v_radial_surf_ml.flatten(), 200, density=True, histtype='step')
            #plt.hist(v_radial_surf_ml.flatten(), 500, density=True, histtype='step')
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
        v_radial_ests = -ati_phases/tau_ati[1]/(k0*2.)

        # ESTIMATE HORIZONTAL VELOCITY
        v_horizo_ests = -ati_phases/tau_ati[1]/(k0*2.)/np.sin(inc_angle)

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

    NRCS_est_avg = 10*np.log10(np.mean(np.mean(i_all[:, :, az_guard:-az_guard, 5:-5], axis=-1), axis=-1))
    output.write('--------------------------------------------\n')
    for pind in range(npol):
        output.write("%s Polarization\n" % polt[pind])
        output.write('Estimated mean NRCS = %5.2f\n' % NRCS_est_avg[0, pind])
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
        plt.hist(v_radial_surf.flatten(), 200, density=True, histtype='step',
                 label='True')
        for pind in range(npol):
            plt.hist(v_radial_ests[pind].flatten(), 200, density=True,
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

    # Save some statistics to npz file
    #
    if num_ch > 1:
        filenpz = os.path.join(os.path.dirname(output_file), 'ati_stats.npz')
        # Mean coh
        cohs = np.array(cohs)[:, az_guard:-az_guard, 5:-5]

        np.savez(filenpz,
                 nrcs=NRCS_est_avg,
                 v_r_dop=np.mean(np.mean(v_radial_ests, axis=-1), axis=-1),
                 v_r_surf=v_radial_surf_mean,
                 v_r_surf_std=v_radial_surf_std,
                 coh_mean=np.mean(np.mean(cohs, axis=-1), axis=-1),
                 abscoh_mean=np.mean(np.mean(np.abs(cohs), axis=-1), axis=-1),
                 coh_lut=coh_lut,
                 pols=polt)
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
