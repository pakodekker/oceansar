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
import json

import numpy as np
# import scipy as sp
import matplotlib.pyplot as plt

from oceansar.memory import memory_limited
from oceansar.utils import geometry as geosar
from drama import utils
from oceansar.utils.geometry import sr_to_geo
from oceansar import ocs_io as tpio
from oceansar.ocs_io import product_metadata as metadata
from oceansar.utils.plotting import finish_plot as finish_figure
from oceansar import constants as const
from oceansar.surfaces import OceanSurface
from oceansar.l2_processing_core import multilook_window, multilook_report


def uwphase(phasor):
    """ This calculates the phase of an array of phasors
    """
    pha = np.angle(phasor)
    nbins = np.min([72, np.ceil(phasor.size / 50).astype(int)])
    pha_hist, h_edgs = np.histogram(pha, bins=nbins, range=(-np.pi, np.pi))
    pha_mod = (h_edgs[pha_hist.argmax()] + h_edgs[pha_hist.argmax()]) / 2
    pha = np.angle(phasor * np.exp(-1j * pha_mod)) + pha_mod
    return pha


@memory_limited
def insar_process(cfg_file, proc_output_file, ocean_file, output_file, *, plot_show=False):

    """Coregister and multilook SLC channels into an InSAR L1b product.

    Parameters
    ----------
    cfg_file : str or path-like
        Configuration supplying insar controls. Geometry, timing, mode and
        polarization are inherited from the input SLC. A target
        insar.resolution is in metres (azimuth, ground range); the selected
        taper determines geometry-dependent integer multilook support.
    proc_output_file : str or path-like
        Complex ProcFile SLC with axes (channel, polarization, azimuth, range).
    ocean_file : str or path-like
        Current-format ocean initial conditions used for surface diagnostics.
    output_file : str or path-like
        Destination L1bFile. ml_intensity has axes (channel, polarization,
        azimuth, range). ml_coherence and ml_phase have axes (pair, azimuth,
        range), where coh_lut maps channel/polarization combinations to
        non-redundant pairs. Coherence is dimensionless, phase is in radians,
        and intensity is normalized power. Sampling, window support and the
        region's coordinate origin are stored as metadata.

    plot_show : bool, optional
        Display enabled diagnostics even when saving. Unsaved enabled figures
        are displayed automatically, and all figures are closed afterward.

    Returns
    -------
    None
        Write the L1b file and configured diagnostics. If the requested
        region has insufficient extent free of azimuth edge effects,
        return without writing the L1b product.

    Notes
    -----
    Configured phase flattening changes the phase reference used by ATI.
    The output is cropped to the requested scene region within the SLC. ATI
    subsequently trims the multilook support to obtain full-window maps.
    """
    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR InSAR Processor: [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-------------------------------------------------------------------')

    print('Initializing...')

    ## CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    # ATI
    rg_ml = getattr(cfg.insar, "rg_ml", None)
    az_ml = getattr(cfg.insar, "az_ml", None)
    target_resolution = getattr(cfg.insar, "resolution", None)
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
    polt = metadata.polarizations(proc_content.__file__, proc_data.shape[1], cfg)
    inherited_attributes = {key: proc_content.__file__.getncattr(key) for key in
        ('processing_geometry', 'processed_doppler_bandwidth_hz', 'antenna_configuration',
         'burst_image_length_m')
        if key in proc_content.__file__.ncattrs()}
    acquisition_prf = metadata.scalar(proc_content.__file__, 'prf', positive=True)
    sr0 = proc_content.get('sr0')
    #print(f"Read sr0 from proc file: {sr0}")
    az0 = proc_content.get('az0')
    print(f"Read az0 from proc file: {az0}")
    inc_angle = proc_content.get('inc_angle')
    b_ati = proc_content.get('b_ati')
    b_xti = proc_content.get('b_xti')
    f0 = proc_content.get('f0')
    prf = metadata.azimuth_sampling(proc_content.__file__)
    num_ch = proc_data.shape[0]
    if metadata.scalar(proc_content.__file__, 'num_ch', positive=True) != num_ch:
        raise ValueError('SLC num_ch header does not match the channel dimension')
    rg_bw = proc_content.get('rg_bw')
    rg_sampling = proc_content.get('rg_sampling')
    v_ground = proc_content.get('v_ground')
    alt = proc_content.get('orbit_alt')
    if 'v_orbit' in proc_content.__file__.variables:
        v_orbit = np.asarray(proc_content.get('v_orbit')).item()
    else:
        v_orbit = geosar.orbit_to_vel(alt, ground=False)
        print(
            "SLC has no v_orbit metadata; using circular-orbit "
            "approximation: %.3f m/s" % v_orbit)
    if 'sin_az' in proc_content.__file__.variables:
        sin_az = np.asarray(proc_content.get('sin_az')).ravel()
    else:
        sin_az = None
    if 'reramp' in proc_content.__file__.variables:
        reramp = bool(np.asarray(proc_content.get('reramp')).item())
    else:
        reramp = bool(getattr(cfg.processing, 'reramp', True))
    operation_mode = metadata.acquisition_mode(proc_content.__file__, cfg)
    inc_angle = np.deg2rad(proc_content.get('inc_angle'))
    # print(f"Read inc_angle from proc file: {np.rad2deg(inc_angle)}")
    proc_content.close()
    if operation_mode == 'scansar' and reramp:
        raise ValueError(
            "ScanSAR InSAR coregistration requires reramp=False")

    ## CALCULATE PARAMETERS
    if v_ground == 'auto':
        v_ground = geosar.orbit_to_vel(
            alt, ground=True, inc=inc_angle)
    k0 = 2.*np.pi*f0/const.c
    print("Orbital velocity used for coregistration: %.3f m/s" % v_orbit)


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
    plot_path = os.path.join(os.path.dirname(output_file), plot_path)
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    # SURFACE VELOCITIES
    grg_grid_spacing = (const.c/2./rg_sampling/np.sin(inc_angle))
    az_grid_spacing = (v_ground/prf)
    spacing = np.array([az_grid_spacing, grg_grid_spacing], dtype=float)
    counts, effective_resolution = multilook_window(
        spacing, ml_win, resolution=target_resolution,
        window_samples=(az_ml, rg_ml), image_shape=proc_data.shape[-2:])
    az_ml, rg_ml = map(int, counts)
    ml_report = multilook_report(counts, spacing, effective_resolution, ml_win)
    if target_resolution is not None:
        ml_report += "  Target resolution (m): " + str(np.broadcast_to(target_resolution, (2,)).tolist()) + "\n"
    print(ml_report)
    # Match the physical averaging width on the ocean grid, independently per axis.
    surface_counts, _ = multilook_window(
        (surface.dy, surface.dx), ml_win, resolution=effective_resolution,
        image_shape=surface.Vx.shape)
    surface_az_ml, surface_rg_ml = map(int, surface_counts)

    # SURFACE RADIAL VELOCITY
    #print("current vector: %f, %f" % (surface.current[0], surface.current[1]))
    v_radial_surf = surface.Vx*np.sin(inc_angle) - surface.Vz*np.cos(inc_angle)
    #print("Mean Vx: %f; Mean Vz: %f" % (np.mean(surface.Vx), np.mean(surface.Vz)))
    v_radial_surf_ml = utils.smooth(utils.smooth(v_radial_surf, surface_rg_ml, axis=1, window=ml_win), surface_az_ml, axis=0, window=ml_win)
    v_radial_surf_mean = np.mean(v_radial_surf)
    #print("Mean radial velocity: %f" % (v_radial_surf_mean))
    v_radial_surf_std = np.std(v_radial_surf)
    v_radial_surf_ml_std = np.std(v_radial_surf_ml)

    # SURFACE HORIZONTAL VELOCITY
    v_horizo_surf = surface.Vx
    v_horizo_surf_ml = utils.smooth(utils.smooth(v_horizo_surf, surface_rg_ml, axis=1, window=ml_win), surface_az_ml, axis=0, window=ml_win)
    v_horizo_surf_mean = np.mean(v_horizo_surf)
    v_horizo_surf_std = np.std(v_horizo_surf)
    v_horizo_surf_ml_std = np.std(v_horizo_surf_ml)

    # Expected mean azimuth shift
    # sr0 = geosar.inc_to_sr(inc_angle, alt)
    avg_az_shift = - v_radial_surf_mean / v_ground * sr0
    print("Expected mean azimuth shift: %f m" % (avg_az_shift))
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
    if sin_az is not None and sin_az.size != az_size:
        raise ValueError(
            "SLC sin_az metadata does not match its azimuth dimension")

    # Note: RG is projected, so plots are Ground Range
    rg_min = 0
    rg_max = int(np.ceil(
        rg_span/(const.c/2./rg_sampling/np.sin(inc_angle))))
    az_min = int(np.floor(
        (-az0 - az_span/2. + avg_az_shift)/(v_ground/prf)))
    az_max = int(np.ceil(
        (-az0 + az_span/2. + avg_az_shift)/(v_ground/prf)))
    requested_roi = (az_min, az_max, rg_min, rg_max)
    az_min = int(np.clip(az_min, 0, az_size))
    az_max = int(np.clip(az_max, 0, az_size))
    rg_min = int(np.clip(rg_min, 0, rg_size))
    rg_max = int(np.clip(rg_max, 0, rg_size))
    roi_az_size = az_max - az_min
    roi_rg_size = rg_max - rg_min
    if roi_az_size <= 0 or roi_rg_size <= 0:
        raise ValueError(
            "The requested InSAR region does not overlap the focused SLC")
    clipped_roi = (az_min, az_max, rg_min, rg_max)
    if clipped_roi != requested_roi:
        print(
            "Clipped InSAR ROI to focused SLC bounds: "
            "azimuth [%d:%d], range [%d:%d]"
            % clipped_roi)
    az_guard = int(std_az_shift / (v_ground / prf))
    if roi_az_size < (2 * az_guard - 10):
        print('Not enough edge-effect free image')
        return

    # b_ati is the receive-antenna separation. In quasi-monostatic SAR the
    # effective phase-center displacement is half this baseline.
    inter_chan_shift_dist = 0.5 * b_ati / (v_orbit/prf)
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
        if operation_mode == 'scansar':
            # ScanSAR uses a common aperture center, so the along-track
            # baseline phase varies systematically across output azimuth.
            print(
                "Applying ScanSAR azimuth flat-earth correction "
                "in deramped coordinates")
            if sin_az is None:
                azimuth_position = (
                    az0 + np.arange(az_size) * v_ground/prf)
                sin_az = azimuth_position / np.sqrt(
                    sr0**2 + azimuth_position**2)
                print(
                    "SLC has no sin_az metadata; using a small-angle "
                    "ScanSAR flattening approximation")
            for ch in range(num_ch):
                proc_data[ch] *= np.exp(
                    1j * k0 * b_ati[ch] * sin_az
                )[np.newaxis, :, np.newaxis]
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
    i_all = i_all.reshape(ch_dim + (roi_az_size, roi_rg_size))
    cohs = np.array(cohs)
    l1b_file = tpio.L1bFile(output_file, 'w', i_all.shape)
    l1b_file.set('ml_intensity', i_all)
    l1b_file.set('ml_coherence', np.abs(cohs))
    l1b_file.set('ml_phase', np.angle(cohs))
    l1b_file.set('coh_lut', coh_lut.reshape((num_ch, npol, num_ch, npol)))
    l1b_file.set('inc_angle', inc_angle)
    l1b_file.set('f0', f0)
    l1b_file.set('num_ch', num_ch)
    metadata.write_sampling(l1b_file.__file__, prf, acquisition_prf)
    l1b_file.__file__.setncatts(inherited_attributes)
    l1b_file.__file__.acquisition_mode = operation_mode
    l1b_file.__file__.polarizations = json.dumps(polt)
    l1b_file.set('v_ground', v_ground)
    l1b_file.set('v_orbit', v_orbit)
    l1b_file.set('orbit_alt', alt)
    l1b_file.set('sr0', sr0)
    l1b_file.set('rg_sampling', rg_sampling)
    # l1b_file.set('rg_bw', rg_bw)
    l1b_file.set('b_ati', b_ati)
    l1b_file.set('b_xti', b_xti)
    l1b_file.set('rg_ml', rg_ml)
    l1b_file.set('az_ml', az_ml)
    l1b_file.__file__.setncatts({
        'ml_window': ml_win,
        'ml_grid_origin_m': [float(az0 + az_min * spacing[0]), float(rg_min * spacing[1])],
        'ml_source_roi_samples': [az_min, az_max, rg_min, rg_max],
        'ml_source_slc': os.path.abspath(proc_output_file),
        'ml_pixel_spacing_m': spacing,
        'ml_window_support_m': counts * spacing,
        'ml_effective_resolution_m': effective_resolution,
        'ml_axis_order': 'azimuth, ground range',
        'ml_resolution_definition': 'sqrt(12 * weighted_variance + pixel_spacing**2)',
    })
    if target_resolution is not None:
        l1b_file.__file__.setncattr('ml_target_resolution_m',
                                  np.broadcast_to(target_resolution, (2,)))
    l1b_file.close()

    image_extent = [(rg_min-.5)*spacing[1], (rg_max-.5)*spacing[1],
                    az0+(az_min-.5)*spacing[0], az0+(az_max-.5)*spacing[0]]
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

        finish_figure(plt.gcf(), os.path.join(plot_path, 'plot_surface.' + plot_format),
                      save=plot_save, show=plot_show, bbox_inches='tight')


    # PROCESSED AMPLITUDE
    if plot_proc_ampl:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'amp_dB_' + polt[pind]+
                         '.' + plot_format)
            plt.figure()
            plt.imshow(utils.db(i_all[0, pind]), aspect='equal',
                       origin='lower',
                       vmin=utils.db(np.max(i_all[0, pind]))-20,
                       extent=image_extent, interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            finish_figure(plt.gcf(), save_path, save=plot_save, show=plot_show)

            save_path = (plot_path + os.sep + 'amp_' + polt[pind]+
                         '.' + plot_format)
            int_img = (i_all[0, pind])**0.5
            vmin = np.mean(int_img) - 3 * np.std(int_img)
            vmax = np.mean(int_img) + 3 * np.std(int_img)
            plt.figure()
            plt.imshow(int_img, aspect='equal',
                       origin='lower',
                       vmin=vmin, vmax=vmax,
                       extent=image_extent, interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            finish_figure(plt.gcf(), save_path, save=plot_save, show=plot_show)

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
                       extent=image_extent,
                       cmap='bone')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("ATI Coherence")
            # plt.colorbar()
            finish_figure(plt.gcf(), save_path, save=plot_save, show=plot_show)

    if num_ch > 1 and (plot_coh or plot_coh_all):
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
                   extent=image_extent,
                   cmap='hsv')
        plt.xlabel('Ground range [m]')
        plt.ylabel('Azimuth [m]')
        plt.title("InSAR Phase")
        plt.colorbar()
        finish_figure(plt.gcf(), save_path, save=plot_save, show=plot_show)



    if npol_ == 4:  # Bypass this for now
        # Cross pol interferogram
        coh_ind = coh_lut[(0, 1)]
        save_path = (plot_path + os.sep + 'POL_coh_' +
                     polt[0] + polt[1] +
                     '.' + plot_format)
        utils.image(np.abs(cohs[coh_ind]), max=1, min=0, aspect='equal',
                    cmap='gray', extent=image_extent,
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
                    cmap=utils.bwr_cmap, extent=image_extent,
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
