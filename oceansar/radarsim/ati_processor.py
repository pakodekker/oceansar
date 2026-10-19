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

from oceansar.memory import memory_limited
from oceansar.utils import geometry as geosar
from drama import utils
from oceansar import ocs_io as tpio
from oceansar.ocs_io import product_metadata as metadata
from oceansar import constants as const
from oceansar.surfaces import OceanSurface
from oceansar.l2_processing_core import multilook_window, multilook_report
from oceansar.radarsim.ati_pairs import estimate_ati_pairs, uwphase


@memory_limited
def ati_process(cfg_file, insar_output_file, ocean_file, output_file, *,
                result_file=None, plot_show=False, plot_bounds=None):
    """Estimate ATI Doppler and radial velocity for all non-redundant antenna pairs.

    Parameters
    ----------
    cfg_file : str or path-like
        Configuration defining radar geometry, ati multilook resolution in
        metres (azimuth, ground range), report paths and diagnostic switches.
    insar_output_file : str or path-like
        InSAR L1bFile containing multilooked intensities, coherence magnitudes,
        phases in radians, pair lookup table, baselines and sampling metadata.
    ocean_file : str or path-like
        Current-format ocean initial conditions for reference surface statistics
        and configured diagnostics.
    output_file : str or path-like
        Destination plain-text statistics report. ati_stats.npz is also written
        in the same directory.
    result_file : str or path-like, optional
        Destination h5netcdf map dataset. Defaults to sim.ati_map_file
        (ati_maps.nc when unset) beside output_file.
    plot_show : bool, optional
        Display enabled figures even when ati.plot_save is True. Figures are
        closed after optional saving and display. With plot_save=False,
        enabled figures are displayed regardless of this argument.
    plot_bounds : dict, optional
        Display limits with 'range' and 'azimuth' entries, each a (min, max)
        pair in metres. These limits affect figures, not saved map values.

    Returns
    -------
    xarray.Dataset or None
        Maps on the valid full-window region, with coordinates in metres.
        pair_dc/pair_radial_velocity/pair_phase/pair_coherence have axes
        (pair, polarization, azimuth, range); lag_dc and lag_radial_velocity
        average finite estimates by channel-index lag. Frequencies are in Hz,
        radial velocities in m/s and phases in radians. Spatial means and
        population standard deviations accompany pair and lag maps. The
        dc/radial_velocity/phase aliases select pair 0-1. Return None if fewer
        than two channels are available.

    Notes
    -----
    Each pair uses its signed time separation baseline / (2 * orbital_speed).
    Radial velocity is -c * Doppler / (2 * carrier_frequency). Phase is locally
    unwrapped around its mode; no ambiguity resolution across lags is applied.
    The input InSAR phase reference, including flattening, is retained.
    """

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR ATI Processor: [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-------------------------------------------------------------------')

    print('Initializing...')

    ## CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    # SAR
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

    def finish_plot(save_path, **save_kwargs):
        figure = plt.gcf()
        if plot_save:
            figure.savefig(save_path, **save_kwargs)
        if plot_show or not plot_save:
            plt.show()
        plt.close(figure)

    # PROCESSED InSAR L1b DATA
    insar_data = tpio.L1bFile(insar_output_file, 'r')
    i_all = insar_data.get('ml_intensity')
    polt = metadata.polarizations(insar_data.__file__, i_all.shape[1], cfg)
    cohs = insar_data.get('ml_coherence') * np.exp(1j * insar_data.get('ml_phase'))
    coh_lut = insar_data.get('coh_lut')
    sr0 = insar_data.get('sr0')
    inc_angle = insar_data.get('inc_angle')
    b_ati = insar_data.get('b_ati')
    b_xti = insar_data.get('b_xti')
    f0 = insar_data.get('f0')
    az_sampling = metadata.azimuth_sampling(insar_data.__file__)
    acquisition_prf = (metadata.scalar(insar_data.__file__, 'prf', positive=True)
                       if 'prf' in insar_data.__file__.variables else az_sampling)
    operation_mode = metadata.acquisition_mode(insar_data.__file__, cfg)
    num_ch = insar_data.get('num_ch')
    rg_sampling = insar_data.get('rg_sampling')
    v_ground = insar_data.get('v_ground')
    alt = insar_data.get('orbit_alt')
    if 'v_orbit' in insar_data.__file__.variables:
        v_orbit = np.asarray(insar_data.get('v_orbit')).item()
    else:
        v_orbit = geosar.orbit_to_vel(alt, ground=False)
        print(
            "InSAR file has no v_orbit metadata; using circular-orbit "
            "approximation: %.3f m/s" % v_orbit)
    # inc_angle = np.deg2rad(insar_data.get('inc_angle'))
    # print("Incidence angle: %f deg" % (np.rad2deg(inc_angle)))
    rg_ml = insar_data.get('rg_ml')
    az_ml = insar_data.get('az_ml')
    ml_win = getattr(insar_data.__file__, 'ml_window',
                     getattr(getattr(cfg, 'insar', None), 'ml_win', ml_win))
    spacing = np.array([v_ground / az_sampling,
                        const.c / (2 * rg_sampling * np.sin(inc_angle))], dtype=float)
    counts, effective_resolution = multilook_window(
        spacing, ml_win, window_samples=(az_ml, rg_ml))
    ml_report = 'Inherited InSAR ' + multilook_report(
        counts, spacing, effective_resolution, ml_win)
    target_resolution = getattr(insar_data.__file__, 'ml_target_resolution_m', None)
    if target_resolution is not None:
        ml_report += '  Target resolution (m): ' + str(np.asarray(target_resolution).tolist()) + '\n'
    print(ml_report)
    grid_origin = getattr(insar_data.__file__, 'ml_grid_origin_m', None)
    insar_data.close()

    # CALCULATE PARAMETERS
    print("Orbital velocity used for ATI timing: %.3f m/s" % v_orbit)

    # OCEAN SURFACE
    surface = OceanSurface()
    surface.load(ocean_file, compute=['D', 'V'])
    surface.t = 0.

    # OUTPUT FILE
    output = open(output_file, 'w')
    output.write(ml_report + '\n')

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
    az_grid_spacing = (v_ground/az_sampling)
    surface_counts, _ = multilook_window(
        (surface.dy, surface.dx), ml_win, resolution=effective_resolution,
        image_shape=surface.Vx.shape)
    surface_az_ml, surface_rg_ml = map(int, surface_counts)

    # SURFACE RADIAL VELOCITY
    v_radial_surf = surface.Vx*np.sin(inc_angle) - surface.Vz*np.cos(inc_angle)
    v_radial_surf_ml = utils.smooth(utils.smooth(v_radial_surf, surface_rg_ml, axis=1, window=ml_win), surface_az_ml, axis=0, window=ml_win)
    v_radial_surf_mean = np.mean(v_radial_surf)
    v_radial_surf_std = np.std(v_radial_surf)
    v_radial_surf_ml_std = np.std(v_radial_surf_ml)

    # SURFACE HORIZONTAL VELOCITY
    v_horizo_surf = surface.Vx
    v_horizo_surf_ml = utils.smooth(utils.smooth(v_horizo_surf, surface_rg_ml, axis=1, window=ml_win), surface_az_ml, axis=0, window=ml_win)
    v_horizo_surf_mean = np.mean(v_horizo_surf)
    v_horizo_surf_std = np.std(v_horizo_surf)
    v_horizo_surf_ml_std = np.std(v_horizo_surf_ml)

    # Expected mean azimuth shift
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

    # Discard partial windows at the ROI boundary before plotting or statistics.
    # Even-length centered convolutions have one more sample on the left.
    left = counts // 2
    right = counts - 1 - left
    left[0] = max(left[0], az_guard)
    right[0] = max(right[0], az_guard)
    shape = np.array(i_all.shape[-2:])
    stop = shape - right
    if np.any(stop <= left):
        raise ValueError('InSAR ROI has no full multilook windows; reduce resolution or enlarge the scene')
    interior = (slice(int(left[0]), int(stop[0])), slice(int(left[1]), int(stop[1])))
    if grid_origin is None:
        grid_origin = (-surface.Ly / 2, 0.)
        origin_source = 'legacy assumed scene origin; regenerate InSAR for exact coordinates'
        print(origin_source)
    else:
        origin_source = 'InSAR ROI origin from focused SLC'
    # Match the actual kernel midpoint, including the half-pixel offset of even windows.
    centre_offset = (counts - 1) / 2 - counts // 2
    axes = [float(origin) + (np.arange(size) + offset) * dx
            for origin, size, offset, dx in zip(grid_origin, shape, centre_offset, spacing)]
    azimuth, ground_range = axes[0][interior[0]], axes[1][interior[1]]
    i_all = i_all[..., interior[0], interior[1]]
    cohs = cohs[..., interior[0], interior[1]]
    estimate_extent = [ground_range[0] - spacing[1]/2, ground_range[-1] + spacing[1]/2,
                       azimuth[0] - spacing[0]/2, azimuth[-1] + spacing[0]/2]
    output.write(f'Valid window-centre extent (m): {estimate_extent}\n')

    ati = None
    if num_ch > 1:
        ati = estimate_ati_pairs(cohs, coh_lut, b_ati, float(v_orbit), float(f0),
                                 polt, azimuth, ground_range)
        ati.attrs.update(source_insar=os.path.abspath(insar_output_file),
            acquisition_mode=operation_mode, az_sampling=float(az_sampling),
            prf=float(acquisition_prf),
            phase_reference='InSAR phase as stored, including configured flattening',
            coordinate_origin_source=origin_source,
            window_samples=counts.tolist(), pixel_spacing_m=spacing.tolist(),
            effective_resolution_m=effective_resolution.tolist(),
            window_support_m=(counts * spacing).tolist(),
            valid_roi_samples=[int(left[0]), int(stop[0]), int(left[1]), int(stop[1])])


    matrix_plot = ati is not None and any((plot_proc_ampl, plot_coh, plot_coh_all,
                                          plot_ati_phase, plot_ati_phase_all))
    if ati is not None:
        ati['intensity'] = (('channel', 'polarization', 'azimuth', 'range'), i_all)
        ati = ati.assign_coords(channel=np.arange(i_all.shape[0]))
        ati.intensity.attrs.update(units='m2/m2', long_name='Multilooked channel intensity')

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

        finish_plot(plot_path + os.sep + 'plot_surface.' + plot_format,
                    bbox_inches='tight')


    # PROCESSED AMPLITUDE
    if plot_proc_ampl and not matrix_plot:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'amp_dB_' + polt[pind]+
                         '.' + plot_format)
            plt.figure()
            plt.imshow(utils.db(i_all[0, pind]), aspect='equal',
                       origin='lower',
                       vmin=utils.db(np.max(i_all[0, pind]))-30,
                       extent=estimate_extent, interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            finish_plot(save_path)
            save_path = (plot_path + os.sep + 'amp_' + polt[pind]
                         + '.' + plot_format)
            int_img = (i_all[0, pind])**0.5
            vmin = np.mean(int_img) - 3 * np.std(int_img)
            vmax = np.mean(int_img) + 3 * np.std(int_img)
            plt.figure()
            plt.imshow(int_img, aspect='equal',
                       origin='lower',
                       vmin=vmin, vmax=vmax,
                       extent=estimate_extent, interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            finish_plot(save_path)

    if matrix_plot:
        from oceansar.radarsim.doppler_plots import plot_ati_matrix
        plot_ati_matrix(ati, plot_path if plot_save else None,
                        show=plot_show or not plot_save, plot_format=plot_format, bounds=plot_bounds)

    if ati is not None:
        # Keep the original adjacent-pair statistics alongside all-pair products.
        doppler_hz = ati.dc.values
        radial_velocity = ati.radial_velocity.values
        if result_file is None:
            result_file = os.path.join(os.path.dirname(output_file),
                getattr(getattr(cfg, 'sim', None), 'ati_map_file', 'ati_maps.nc'))
        os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
        ati.to_netcdf(result_file, engine='h5netcdf')
        pair_summary = ati[['pair_dc_mean', 'pair_dc_std', 'pair_radial_velocity_mean', 'pair_radial_velocity_std']].to_dataframe()
        lag_summary = ati[['lag_dc_mean', 'lag_dc_std', 'lag_radial_velocity_mean', 'lag_radial_velocity_std', 'lag_pair_count']].to_dataframe()
        output.write('ATI PER-PAIR SPATIAL MEAN AND STD (Hz, m/s)\n' + pair_summary.to_string() + '\n\n')
        output.write('ATI LAG-AVERAGED SPATIAL MEAN AND STD (Hz, m/s)\n' + lag_summary.to_string() + '\n\n')
        print('ATI lag averages:')
        print(lag_summary.to_string())
        if plot_ati_phase or plot_ati_phase_all or plot_vel or plot_show:
            from oceansar.radarsim.doppler_plots import plot_ati
            plot_ati(ati, plot_path if plot_save else None, show=plot_show,
                     plot_format=plot_format, bounds=plot_bounds)

        output.write('--------------------------------------------\n')
        output.write('SURFACE RADIAL VELOCITY - NO SMOOTHING\n')
        output.write('MEAN(SURF. V) = %.4f\n' % v_radial_surf_mean)
        output.write('STD(SURF. V) = %.4f\n' % v_radial_surf_std)
        output.write('--------------------------------------------\n\n')

        output.write('--------------------------------------------\n')
        output.write('SURFACE RADIAL VELOCITY - SMOOTHING (OCEAN GRID WIN. SIZE=%dx%d)\n' % (surface_az_ml, surface_rg_ml))
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

            finish_plot(plot_path + os.sep + 'TRUE_radial_vel_hist.' + plot_format)

            plt.figure()
            plt.hist(v_radial_surf_ml.flatten(), 200, density=True, histtype='step')
            #plt.hist(v_radial_surf_ml.flatten(), 500, density=True, histtype='step')
            plt.grid(True)
            plt.xlim([-np.abs(v_radial_surf_mean) - 4.*v_radial_surf_std, np.abs(v_radial_surf_mean) + 4.* v_radial_surf_std])
            plt.xlabel('Radial velocity [m/s]')
            plt.ylabel('PDF')
            plt.title('Surface velocity (low pass filtered)')

            finish_plot(plot_path + os.sep + 'TRUE_radial_vel_ml_hist.' + plot_format)

        if plot_vel:
            from oceansar.radarsim.doppler_plots import symmetric_limit
            for values, title, filename in (
                    (v_radial_surf, 'Surface Radial Velocity', 'TRUE_radial_vel.'),
                    (v_radial_surf_ml, 'Surface Radial Velocity (low pass filtered)', 'TRUE_radial_vel_ml.')):
                plt.figure()
                limit = symmetric_limit(values)
                image = plt.imshow(values, aspect='equal', origin='lower', cmap='bwr',
                                   extent=[0., rg_span, 0., az_span],
                                   vmin=-limit, vmax=limit)
                plt.xlabel('Ground range [m]')
                plt.ylabel('Azimuth [m]')
                plt.title(title)
                plt.colorbar(image, label='Radial velocity (m/s)')
                finish_plot(plot_path + os.sep + filename + plot_format)

        ##  ESTIMATED VELOCITIES

        # Note: plot limits are taken from surface calculations to keep the same ranges

        # ESTIMATE RADIAL VELOCITY
        v_radial_ests = ati.radial_velocity.values

        # ESTIMATE HORIZONTAL VELOCITY
        v_horizo_ests = v_radial_ests/np.sin(inc_angle)

        output.write('--------------------------------------------\n')
        output.write('ESTIMATED RADIAL VELOCITY - INHERITED INSAR MULTILOOKING\n')
        for pind in range(npol):
            output.write("%s Polarization\n" % polt[pind])
            output.write('MEAN(EST. V) = %.4f\n' % np.mean(v_radial_ests[pind]))
            output.write('STD(EST. V) = %.4f\n' % np.std(v_radial_ests[pind]))
        output.write('--------------------------------------------\n\n')

        output.write('--------------------------------------------\n')
        output.write('ESTIMATED RADIAL VELOCITY - ADDITIONAL BOXCAR SMOOTHING (WIN. SIZE=%dx%d)\n' % (az_ml, rg_ml))
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
        output.write('ESTIMATED HORIZONTAL VELOCITY - INHERITED INSAR MULTILOOKING\n')
        for pind in range(npol):
            output.write("%s Polarization\n" % polt[pind])
            output.write('MEAN(EST. V) = %.4f\n' % np.mean(v_horizo_ests[pind]))
            output.write('STD(EST. V) = %.4f\n' % np.std(v_horizo_ests[pind]))
        output.write('--------------------------------------------\n\n')

    # Processed NRCS

    NRCS_est_avg = 10*np.log10(np.mean(np.mean(i_all, axis=-1), axis=-1))
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

        finish_plot(plot_path + os.sep + 'ATI_radial_vel_hist.' + plot_format)

    # Save some statistics to npz file
    #
    if num_ch > 1:
        filenpz = os.path.join(os.path.dirname(output_file), 'ati_stats.npz')
        # Mean coh
        cohs = np.array(cohs)

        np.savez(filenpz,
                 nrcs=NRCS_est_avg,
                 v_r_dop=np.mean(np.mean(v_radial_ests, axis=-1), axis=-1),
                 doppler_hz=doppler_hz, radial_velocity=radial_velocity,
                 doppler_mean_hz=ati.dc_mean.values,
                 pair_doppler_hz=ati.pair_dc.values,
                 pair_radial_velocity=ati.pair_radial_velocity.values,
                 pair_doppler_mean_hz=ati.pair_dc_mean.values,
                 pair_doppler_std_hz=ati.pair_dc_std.values,
                 pair_radial_velocity_std=ati.pair_radial_velocity_std.values,
                 pair_radial_velocity_mean=ati.pair_radial_velocity_mean.values,
                 pairs=ati.pair.values, antenna_lag=ati.antenna_lag.values,
                 pair_time_separation_s=ati.time_separation_s.values,
                 lag_doppler_hz=ati.lag_dc.values,
                 lag_radial_velocity=ati.lag_radial_velocity.values,
                 lag_doppler_mean_hz=ati.lag_dc_mean.values,
                 lag_doppler_std_hz=ati.lag_dc_std.values,
                 lag_radial_velocity_std=ati.lag_radial_velocity_std.values,
                 lag_radial_velocity_mean=ati.lag_radial_velocity_mean.values,
                 lag_pair_count=ati.lag_pair_count.values, lags=ati.lag.values,
                 azimuth_m=azimuth, ground_range_m=ground_range,
                 v_r_surf=v_radial_surf_mean,
                 v_r_surf_std=v_radial_surf_std,
                 coh_mean=np.mean(np.mean(cohs, axis=-1), axis=-1),
                 abscoh_mean=np.mean(np.mean(np.abs(cohs), axis=-1), axis=-1),
                 coh_lut=coh_lut,
                 pols=polt,
                 ml_window=ml_win, ml_window_samples=counts,
                 ml_pixel_spacing_m=spacing,
                 ml_effective_resolution_m=effective_resolution,
                 ml_window_support_m=counts * spacing)
    print('----------------------------------------')
    print(time.strftime("ATI Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('----------------------------------------')
    return ati


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-p', '--proc_file')
    parser.add_argument('-s', '--ocean_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    ati_process(args.cfg_file, args.proc_file, args.ocean_file, args.output_file)
