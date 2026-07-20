#!/usr/bin/env python

""" ============================================
    SRP: SAR Raw data Processor (:mod:`srp`)
    ============================================

    Script to process SAR RAW data

    **Arguments**
        * -c, --cfg_file: Configuration file
        * -r, --raw_file: Raw data file
        * -o, --output_file: Output file

"""

import os
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as scipy_signal

from drama.geo.geo_history import GeoHistory
from drama.performance.sar.sar_performance_common import calc_analysis_time
from oceansar.utils import geometry as geo
from oceansar import utils
from oceansar import ocs_io as tpio
from oceansar import constants as const
from oceansar.radarsim.antenna import sinc_1tx_nrx, sinc_bp


def make_geohistory(cfg, inc_angle, fit_half_span=1.0):
    """Create a GeoHistory object covering the local processing interval."""
    alt = cfg.orbit.Horb
    t_analysis = max(
        2.0 * fit_half_span + 2.0,
        calc_analysis_time(
            alt, inc_angle, cfg.sar.f0, cfg.sar.prf, n_amb=1))
    return GeoHistory(
        cfg,
        latitude=10,
        inc_range=(np.degrees(
            np.array([inc_angle, inc_angle + np.radians(30.0)]))
            + np.array([-3.0, 3.0])),
        inc_swth=np.degrees(inc_angle) + np.array([-1.0, 1.0]),
        n_la_pts=800,
        aei=None,
        t_analysis=t_analysis)


def estimate_effective_velocity(cfg, inc_angle, fit_half_span=1.0, ghist=None):
    """Fit the local GeoHistory range curvature and return its velocity."""
    alt = cfg.orbit.Horb
    if ghist is None:
        ghist = make_geohistory(cfg, inc_angle, fit_half_span=fit_half_span)

    gr0 = geo.inc_to_gr(inc_angle, alt)
    look_angle = np.asarray(
        geo.gr_to_geo(np.array([gr0]), alt)[2]).item()
    fit_time = np.linspace(-fit_half_span, fit_half_span, 9)
    slant_range = ghist.sr_spl(look_angle, fit_time).ravel()
    range_zero = ghist.sr_spl(look_angle, 0.0).item()
    time_squared = fit_time**2
    curvature = np.dot(time_squared, slant_range - range_zero) / np.dot(
        time_squared, time_squared)
    if curvature <= 0:
        raise ValueError(
            "GeoHistory produced non-positive slant-range curvature")
    return np.sqrt(2.0 * range_zero * curvature)


def scene_coordinates(cfg):
    """Return the optimized raw-generator scene coordinates."""
    nx = int(cfg.ocean.Lx/cfg.ocean.dx)
    ny = int(cfg.ocean.Ly/cfg.ocean.dy)
    if cfg.ocean.opt_res:
        nx = int(utils.optimize_fftsize(nx, cfg.ocean.fft_max_prime))
        ny = int(utils.optimize_fftsize(ny, cfg.ocean.fft_max_prime))
    x = np.linspace(-cfg.ocean.Lx/2., cfg.ocean.Lx/2., nx)
    y = np.linspace(-cfg.ocean.Ly/2., cfg.ocean.Ly/2., ny)
    return x, y


def reference_point_geometry(cfg, inc_angle):
    """Return the raw-generator reference-point geometry."""
    alt = cfg.orbit.Horb
    x, y = scene_coordinates(cfg)
    gr0 = geo.inc_to_gr(inc_angle, alt)
    look_near = np.asarray(geo.gr_to_geo(np.array([gr0 + x[0]]), alt)[2]).item()
    look_pt = np.asarray(
        geo.gr_to_geo(np.array([gr0 + x[x.size//2]]), alt)[2]).item()
    return look_near, look_pt, y[y.size//2]


def point_target_geometry(cfg, inc_angle):
    """Return names, look angles, and azimuth positions for the target cross."""
    x, y = scene_coordinates(cfg)
    x_edge = int(round(0.1*(x.size - 1)))
    y_edge = int(round(0.1*(y.size - 1)))
    x_center = x.size//2
    y_center = y.size//2
    y_positions = (
        ('late', y.size - 1 - y_edge),
        ('center', y_center),
        ('early', y_edge),
    )
    x_positions = (
        ('near', x_edge),
        ('center', x_center),
        ('far', x.size - 1 - x_edge),
    )
    gr0 = geo.inc_to_gr(inc_angle, cfg.orbit.Horb)
    targets = []
    for y_name, y_index in y_positions:
        for x_name, x_index in x_positions:
            name = ('center' if y_name == x_name == 'center'
                    else '%s_%s' % (y_name, x_name))
            look = np.asarray(geo.gr_to_geo(
                np.array([gr0 + x[x_index]]), cfg.orbit.Horb)[2]).item()
            targets.append((name, look, y[y_index]))
    return targets


def geohistory_reference_filter(ghist, look_angle, fa, f0):
    """Return GeoHistory RCMC and azimuth compression at one look angle."""
    fa_order = np.argsort(fa)[::-1]
    fa_sorted = fa[fa_order]
    t_dop, _, _, _, _ = ghist.Doppler2tuv(
        look_angle, fa_sorted, f0)
    t_zero_dop, _, _, _, _ = ghist.Doppler2tuv(
        look_angle, np.array([0.0]), f0)
    t_dop = np.asarray(t_dop).ravel()
    t_zero_dop = np.asarray(t_zero_dop).item()
    sr_zero_dop = ghist.sr_spl(look_angle, t_zero_dop).item()
    sr_dop = ghist.sr_spl(look_angle, t_dop).ravel()
    range_migration_sorted = sr_dop - sr_zero_dop

    wavelength = const.c/f0
    phase_sorted = (4*np.pi/wavelength * range_migration_sorted
                    + 2*np.pi*fa_sorted*(t_dop - t_zero_dop))

    range_migration = np.empty_like(range_migration_sorted)
    range_migration[fa_order] = range_migration_sorted
    phase = np.empty_like(phase_sorted)
    phase[fa_order] = phase_sorted
    return range_migration, phase, sr_zero_dop, t_zero_dop


def geohistory_range_azimuth_filter(ghist, slant_range, fa, f0):
    """Return azimuth compression phase for every slant-range bin."""
    look_grid = ghist._la_vector
    sr_grid = ghist.sr_spl(look_grid, 0.0).ravel()
    look_angles = np.interp(slant_range, sr_grid, look_grid)

    fa_order = np.argsort(fa)[::-1]
    fa_sorted = fa[fa_order]
    t_dop, _, _, _, _ = ghist.Doppler2tuv(
        look_angles, fa_sorted, f0)
    t_zero_dop, _, _, _, _ = ghist.Doppler2tuv(
        look_angles, np.array([0.0]), f0)
    t_dop = np.asarray(t_dop)
    t_zero_dop = np.asarray(t_zero_dop).reshape((-1, 1))
    sr_zero_dop = ghist.sr_spl.ev(look_angles, t_zero_dop[:, 0])
    sr_dop = ghist.sr_spl.ev(look_angles[:, np.newaxis], t_dop)
    range_migration = sr_dop - sr_zero_dop[:, np.newaxis]

    wavelength = const.c/f0
    phase_sorted = (
        4*np.pi/wavelength*range_migration
        + 2*np.pi*fa_sorted[np.newaxis, :]*(t_dop - t_zero_dop))
    phase = np.empty_like(phase_sorted)
    phase[:, fa_order] = phase_sorted
    return phase.T


def point_target_azimuth_profile(data, expected_az, expected_rg,
                                 oversample=16, span=10):
    """Return an oversampled, phase-compensated point-target profile."""
    az_center = int(np.round(expected_az))
    rg_center = int(np.round(expected_rg))
    az_slice = slice(max(0, az_center - 8),
                     min(data.shape[1], az_center + 9))
    rg_slice = slice(max(0, rg_center - 4),
                     min(data.shape[2], rg_center + 5))
    local_image = np.abs(data[0, az_slice, rg_slice])
    local_peak = np.unravel_index(np.argmax(local_image), local_image.shape)
    az_peak = az_slice.start + local_peak[0]
    rg_peak = rg_slice.start + local_peak[1]

    profile = data[0, :, rg_peak]
    profile_oversampled = scipy_signal.resample(
        profile, profile.size*oversample)
    peak_search = slice(max(0, (az_peak - 2)*oversample),
                        min(profile_oversampled.size,
                            (az_peak + 3)*oversample))
    peak_index = (peak_search.start
                  + np.argmax(np.abs(profile_oversampled[peak_search])))
    peak_value = profile_oversampled[peak_index]
    profile_oversampled *= np.exp(-1j*np.angle(peak_value))/np.abs(peak_value)

    azimuth_samples = (np.arange(profile_oversampled.size)/oversample
                       - expected_az)
    plot_samples = np.abs(azimuth_samples) <= span/2
    return (azimuth_samples[plot_samples],
            profile_oversampled[plot_samples],
            peak_index/oversample, rg_peak)


def plot_point_target_azimuth_grid(data, expected_targets, channel,
                                   plot_path, plot_format,
                                   oversample=16, span=10):
    """Plot the 3x3 target-grid azimuth responses for one channel."""
    fig, axs = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (target_name, expected_az, expected_rg) in zip(
            axs.ravel(), expected_targets):
        azimuth_samples, profile, peak_az, peak_rg = (
            point_target_azimuth_profile(
                data, expected_az, expected_rg,
                oversample=oversample, span=span))
        ax.plot(azimuth_samples, np.real(profile),
                label='real, peak phase removed')
        ax.plot(azimuth_samples, np.abs(profile), '--', label='magnitude')
        ax.plot(azimuth_samples, np.imag(profile), ':', label='imaginary')
        ax.set_title(target_name.replace('_', ' / '))
        ax.set_xlim(-span/2, span/2)
        ax.grid(True)
        print(
            "Point-target peak [%s, channel %d]: az=%.3f, range=%d" %
            (target_name, channel, peak_az, peak_rg))

    for ax in axs[-1, :]:
        ax.set_xlabel('Azimuth offset [original samples]')
    for ax in axs[:, 0]:
        ax.set_ylabel('Normalized amplitude')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle('Point-target azimuth responses, channel %d' % channel,
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(
        plot_path, 'plot_point_target_azimuth_grid_%d.%s' %
        (channel, plot_format)))
    plt.close(fig)


def check_reference_range_history(cfg, ghist, inc_angle, raw_sr0, az0,
                                  prf, v_ground, f0, sr_pt, az_size,
                                  v_eff, plot_path=None, plot_format='png',
                                  plot_save=False):
    """Print consistency diagnostics for the saved reference range history."""
    raw_sr0 = np.asarray(raw_sr0).item()
    az0 = np.asarray(az0).item()
    sr_pt = np.asarray(sr_pt)
    look_near, look_pt, y_pt = reference_point_geometry(cfg, inc_angle)
    range_ref = ghist.sr_spl(look_near, 0.0).item()
    time_raw = az0/v_ground + np.arange(sr_pt.size)/prf - y_pt/v_ground
    sr_geom = ghist.sr_spl(look_pt, time_raw).ravel() - range_ref
    sr_diff = sr_pt - sr_geom
    print(
        "sr_pt GeoHistory check: mean=%+.4e m, rms=%.4e m, max=%.4e m" %
        (np.mean(sr_diff), np.sqrt(np.mean(sr_diff**2)),
         np.max(np.abs(sr_diff))))

    fa = np.fft.fftfreq(az_size, 1/prf)
    fa_order = np.argsort(fa)[::-1]
    fa_sorted = fa[fa_order]
    sr_dop, _, sr_zero_dop, t_zero_dop = geohistory_reference_filter(
        ghist, look_pt, fa, f0)
    sr_dop_sorted = sr_dop[fa_order]
    rcmc_denominator = np.sqrt(
        1 - (fa * (const.c/f0 / 2.) / v_eff)**2.)
    rcmc_fa = raw_sr0 / rcmc_denominator - raw_sr0
    rcmc_fa_pt = sr_zero_dop / rcmc_denominator - sr_zero_dop
    print(
        "GeoHistory zero-Doppler time offset: %+.4e s, range offset: %+.4e m" %
        (t_zero_dop, sr_zero_dop - ghist.sr_spl(look_pt, 0.0).item()))
    valid = (np.isfinite(sr_dop) & np.isfinite(rcmc_fa)
             & np.isfinite(rcmc_fa_pt))
    if np.any(valid):
        rcmc_diff = rcmc_fa[valid] - sr_dop[valid]
        rcmc_diff_pt = rcmc_fa_pt[valid] - sr_dop[valid]
        print(
            "RCMC near-range check: mean=%+.4e m, rms=%.4e m, max=%.4e m" %
            (np.mean(rcmc_diff), np.sqrt(np.mean(rcmc_diff**2)),
             np.max(np.abs(rcmc_diff))))
        print(
            "RCMC point-range check: mean=%+.4e m, rms=%.4e m, max=%.4e m" %
            (np.mean(rcmc_diff_pt), np.sqrt(np.mean(rcmc_diff_pt**2)),
             np.max(np.abs(rcmc_diff_pt))))

    if plot_save and plot_path is not None:
        rcmc_sorted = rcmc_fa[fa_order]
        rcmc_pt_sorted = rcmc_fa_pt[fa_order]
        rcmc_diff_sorted = rcmc_sorted - sr_dop_sorted
        rcmc_diff_pt_sorted = rcmc_pt_sorted - sr_dop_sorted
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(time_raw, sr_pt, label='raw sr_pt')
        axs[0, 0].plot(time_raw, sr_geom, label='GeoHistory')
        axs[0, 0].set_xlabel('Slow time [s]')
        axs[0, 0].set_ylabel('Range history [m]')
        axs[0, 0].set_title('Reference point range history')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(time_raw, sr_diff)
        axs[0, 1].set_xlabel('Slow time [s]')
        axs[0, 1].set_ylabel('Range difference [m]')
        axs[0, 1].set_title('Reference point difference')
        axs[0, 1].grid(True)

        axs[1, 0].plot(fa_sorted, rcmc_sorted, label='near-range RCMC')
        axs[1, 0].plot(fa_sorted, rcmc_pt_sorted,
                       label='point-range RCMC')
        axs[1, 0].plot(fa_sorted, sr_dop_sorted, label='GeoHistory')
        axs[1, 0].set_xlabel('Doppler frequency [Hz]')
        axs[1, 0].set_ylabel('Range migration [m]')
        axs[1, 0].set_title('RCM history')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].plot(fa_sorted, rcmc_diff_sorted, label='near range')
        axs[1, 1].plot(fa_sorted, rcmc_diff_pt_sorted,
                       label='point range')
        axs[1, 1].set_xlabel('Doppler frequency [Hz]')
        axs[1, 1].set_ylabel('Range difference [m]')
        axs[1, 1].set_title('RCM difference')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(
            plot_path, 'plot_range_history_check.%s' % plot_format))
        plt.close(fig)


def sar_focus(cfg_file, raw_output_file, output_file):

    ###################
    # INITIALIZATIONS #
    ###################

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR Processor: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('-------------------------------------------------------------------')

    # CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    # PROCESSING
    az_weighting = cfg.processing.az_weighting
    doppler_bw = cfg.processing.doppler_bw
    plot_format = cfg.processing.plot_format
    plot_tex = cfg.processing.plot_tex
    plot_save = cfg.processing.plot_save
    plot_path = cfg.processing.plot_path
    plot_raw = cfg.processing.plot_raw
    plot_rcmc_dopp = cfg.processing.plot_rcmc_dopp
    plot_rcmc_time = cfg.processing.plot_rcmc_time
    plot_image_valid = cfg.processing.plot_image_valid
    range_dependent_azimuth = (
        cfg.processing.range_dependent_azimuth
        if hasattr(cfg.processing, 'range_dependent_azimuth') else True)
    add_point_target = (cfg.sim.add_point_target
                        if hasattr(cfg.sim, 'add_point_target') else False)

    # SAR
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    num_ch = cfg.sar.num_ch
    alt = cfg.sar.alt
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.sar.rg_bw
    over_fs = cfg.sar.over_fs

    # CALCULATE PARAMETERS
    l0 = const.c / f0
    if v_ground == 'auto':
        v_ground = geo.orbit_to_vel(
            alt, ground=True, inc=np.deg2rad(cfg.sar.inc_angle))
    rg_sampling = rg_bw * over_fs

    # RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    sr0 = raw_file.get('sr0')
    az0 = raw_file.get('az0')
    inc_angle = raw_file.get('inc_angle')
    b_ati = raw_file.get('b_ati')
    b_xti = raw_file.get('b_xti')
    sr_pt = None
    if 'sr_pt' in raw_file.__file__.variables:
        sr_pt = raw_file.get('sr_pt')
    raw_file.close()
    inc_angle_rad = np.asarray(np.deg2rad(inc_angle)).item()
    ghist = make_geohistory(cfg, inc_angle_rad)
    v_eff = estimate_effective_velocity(cfg, inc_angle_rad, ghist=ghist)
    _, look_focus, _ = reference_point_geometry(cfg, inc_angle_rad)
    point_targets = point_target_geometry(cfg, inc_angle_rad)
    print("Effective focusing velocity: %.3f m/s" % v_eff)
    print("Range-dependent azimuth compression: %s" %
          range_dependent_azimuth)

    # OTHER INITIALIZATIONS
    # Create plots directory
    plot_path = os.path.dirname(output_file) + os.sep + plot_path
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    slc = []
    filter_cache = {}

    ########################
    # PROCESSING MAIN LOOP #
    ########################
    for ch in np.arange(num_ch):

        if plot_raw:
            plt.figure()
            plt.imshow(np.real(raw_data[0, ch]),
                       vmin=-np.max(np.abs(raw_data[0, ch])),
                       vmax=np.max(np.abs(raw_data[0, ch])), cmap='gray')
            plt.savefig(plot_path + os.sep + ('plot_raw_real_%d.%s' % (ch, plot_format)))
            plt.close()
            # utils.image(np.real(raw_data[0, ch]), min=-np.max(np.abs(raw_data[0, ch])), max=np.max(np.abs(raw_data[0, ch])), cmap='gray',
            #             aspect=np.float(
            #                 raw_data[0, ch].shape[1]) / np.float(raw_data[0, ch].shape[0]),
            #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
            #             usetex=plot_tex,
            #             save=plot_save, save_path=plot_path + os.sep +
            #             'plot_raw_real_%d.%s' % (ch, plot_format),
            #             dpi=150)
            # utils.image(np.imag(raw_data[0, ch]), min=-np.max(np.abs(raw_data[0, ch])), max=np.max(np.abs(raw_data[0, ch])), cmap='gray',
            #             aspect=np.float(
            #                 raw_data[0, ch].shape[1]) / np.float(raw_data[0, ch].shape[0]),
            #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
            #             usetex=plot_tex,
            #             save=plot_save, save_path=plot_path + os.sep +
            #             'plot_raw_imag_%d.%s' % (ch, plot_format),
            #             dpi=150)
            # utils.image(np.abs(raw_data[0, ch]), min=0, max=np.max(np.abs(raw_data[0, ch])), cmap='gray',
            #             aspect=np.float(
            #                 raw_data[0, ch].shape[1]) / np.float(raw_data[0, ch].shape[0]),
            #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
            #             usetex=plot_tex,
            #             save=plot_save, save_path=plot_path + os.sep +
            #             'plot_raw_amp_%d.%s' % (ch, plot_format),
            #             dpi=150)
            # utils.image(np.angle(raw_data[0, ch]), min=-np.pi, max=np.pi, cmap='gray',
            #             aspect=np.float(
            #                 raw_data[0, ch].shape[1]) / np.float(raw_data[0, ch].shape[0]),
            #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
            #             usetex=plot_tex, save=plot_save,
            #             save_path=plot_path + os.sep +
            #             'plot_raw_phase_%d.%s' % (ch, plot_format),
            #             dpi=150)

        # Optimize matrix sizes
        az_size_orig, rg_size_orig = raw_data[0, ch].shape
        if ch == 0 and sr_pt is not None:
            check_reference_range_history(
                cfg, ghist, inc_angle_rad, sr0, az0, prf, v_ground, f0,
                sr_pt, az_size_orig, v_eff, plot_path=plot_path,
                plot_format=plot_format, plot_save=plot_save)
        optsize = utils.optimize_fftsize(raw_data[0, ch].shape)
        optsize = [raw_data.shape[0], optsize[0], optsize[1]]
        data = np.zeros(optsize, dtype=complex)
        data[:, :raw_data[0, ch].shape[0],
             :raw_data[0, ch].shape[1]] = raw_data[:, ch, :, :]

        az_size, rg_size = data.shape[1:]

        # RCMC Correction
        print('Applying RCMC correction... [Channel %d/%d]' % (ch + 1, num_ch))

        # fr = np.linspace(-rg_sampling/2., rg_sampling/2., rg_size)
        # fr = (np.arange(rg_size) - rg_size / 2) * rg_sampling / rg_size
        # fr = np.roll(fr, int(-rg_size / 2))
        fr = np.fft.fftfreq(rg_size, 1/rg_sampling)
        # fa = (np.arange(az_size) - az_size / 2) * prf / az_size
        # fa = np.roll(fa, int(-az_size / 2))
        fa = np.fft.fftfreq(az_size, 1/prf)
        ## Compensation of ANTENNA PATTERN
        ## FIXME this will not work for a long separation betwen Tx and Rx!!!
        sin_az = fa * l0 / (2 * v_eff)
        if hasattr(cfg.sar, 'ant_L'):
            ant_L = cfg.sar.ant_L
            if cfg.sar.L_total:
                beam_pattern = sinc_1tx_nrx(sin_az, ant_L * num_ch, f0, num_ch, field=True)
            else:
                beam_pattern = sinc_1tx_nrx(sin_az, ant_L, f0, 1, field=True)
        else:
            ant_l_tx = cfg.sar.ant_L_tx
            ant_l_rx = cfg.sar.ant_L_rx
            beam_pattern = (sinc_bp(sin_az, ant_l_tx, f0, field=True)
                            * sinc_bp(sin_az, ant_l_rx, f0, field=True))
        #fa[az_size/2:] = fa[az_size/2:] - prf
        filter_key = (az_size, rg_size)
        if filter_key not in filter_cache:
            rcmc_fa, ph_ac_reference, _, _ = (
                geohistory_reference_filter(ghist, look_focus, fa, f0))
            if range_dependent_azimuth:
                slant_range = (sr0 + np.arange(rg_size)
                               * const.c/(2*rg_sampling))
                ph_ac = geohistory_range_azimuth_filter(
                    ghist, slant_range, fa, f0)
            else:
                ph_ac = ph_ac_reference
            filter_cache[filter_key] = (rcmc_fa, ph_ac)
        rcmc_fa, ph_ac = filter_cache[filter_key]
        #rcmc_fa[:]=0
        data = np.fft.fft(np.fft.fft(data, axis=-1), axis=-2)

#        for i in np.arange(az_size):
#            data[i,:] *= np.exp(1j*2*np.pi*2*rcmc_fa[i]/const.c*fr)
        data = (data * np.exp(4j * np.pi * rcmc_fa.reshape((1, az_size, 1)) /
                              const.c * fr.reshape((1, 1, rg_size))))
        data = np.fft.ifft(data, axis=2)

        if plot_rcmc_dopp:
            plt.figure()
            plt.imshow(np.fft.fftshift(np.abs(data[0]), axes=0), vmax=np.max(np.abs(data)), cmap='gray',
                       origin='lower')
            plt.savefig(plot_path + os.sep + ('plot_rcmc_dopp_%d.%s' % (ch, plot_format)))

        if plot_rcmc_time:
            rcmc_time = np.fft.ifft(data[0], axis=0)[
                :az_size_orig, :rg_size_orig]
            rcmc_time_max = np.max(np.abs(rcmc_time))
            plt.figure()
            plt.imshow(np.real(rcmc_time), vmin=-rcmc_time_max, vmax=rcmc_time_max, cmap='gray',
                       origin='lower')
            plt.savefig(plot_path + os.sep + ('plot_rcmc_time_real_%d.%s' % (ch, plot_format)))

        # Azimuth compression
        print(
            'Applying azimuth compression... [Channel %d/%d]' % (ch + 1, num_ch))

        n_samp = 2 * (int(doppler_bw / (fa[1] - fa[0])) / 2)
        weighting = (az_weighting -
                     (1. - az_weighting) * np.cos(2 * np.pi * np.linspace(0, 1., int(n_samp))))
        # Compensate amplitude loss

        L_win = np.sum(np.abs(weighting)**2) / weighting.size
        weighting /= np.sqrt(L_win)
        if fa.size > n_samp:
            zeros = np.zeros(az_size)
            zeros[0:int(n_samp)] = weighting
            # zeros[:n_samp/2] = weighting[:n_samp/2]
            # zeros[-n_samp/2:] = weighting[-n_samp/2:]
            weighting = np.roll(zeros, int(-n_samp / 2))
        weighting = np.where(np.abs(beam_pattern) > 0, weighting/beam_pattern, 0)
#        for i in np.arange(rg_size):
#            data[:,i] *= np.exp(1j*ph_ac)*weighting
        if ph_ac.ndim == 1:
            azimuth_filter = (
                np.exp(1j*ph_ac)*weighting).reshape((1, az_size, 1))
        else:
            azimuth_filter = (np.exp(1j*ph_ac)[np.newaxis, :, :]
                              * weighting.reshape((1, az_size, 1)))
        data = data*azimuth_filter

        data = np.fft.ifft(data, axis=1)

        print('Finishing... [Channel %d/%d]' % (ch + 1, num_ch))
        # Reduce to initial dimension
        data = data[:, :int(az_size_orig), :int(rg_size_orig)]

        # Removal of non valid samples
        n_val_az_2 = np.floor(
            doppler_bw / 2. / (2. * v_eff**2. / l0 / sr0) * prf / 2.) * 2.
        # data = raw_data[ch, n_val_az_2:(az_size_orig - n_val_az_2 - 1), :]
        data = data[:, int(n_val_az_2):int(az_size_orig - n_val_az_2 - 1), :]
        if add_point_target and plot_save:
            expected_targets = []
            for target_name, target_look, target_y in point_targets:
                target_sr = ghist.sr_spl(target_look, 0.0).item()
                expected_az = (
                    (target_y - az0)*prf/v_ground - n_val_az_2)
                expected_rg = (target_sr - sr0)*2*rg_sampling/const.c
                expected_targets.append(
                    (target_name, expected_az, expected_rg))
            plot_point_target_azimuth_grid(
                data, expected_targets, ch, plot_path, plot_format)
        if plot_image_valid:
            plt.figure()
            plt.imshow(np.abs(data[0]), origin='lower', vmin=0, vmax=np.max(np.abs(data)),
                       aspect=float(rg_size_orig) / float(az_size_orig),
                       cmap='gray')
            plt.xlabel("Range")
            plt.ylabel("Azimuth")
            plt.savefig(os.path.join(
                plot_path, ('plot_image_valid_%d.%s' % (ch, plot_format))))

        slc.append(data)

    # Save processed data
    slc = np.array(slc, dtype=complex)
    print("Shape of SLC: " + str(slc.shape), flush=True)
    proc_file = tpio.ProcFile(output_file, 'w', slc.shape)
    proc_file.set('slc*', slc)
    proc_file.set('inc_angle', inc_angle)
    proc_file.set('f0', f0)
    proc_file.set('num_ch', num_ch)
    proc_file.set('ant_L', ant_l_tx)
    proc_file.set('prf', prf)
    proc_file.set('v_ground', v_ground)
    proc_file.set('az0', az0+n_val_az_2*(v_ground/prf))
    proc_file.set('orbit_alt', alt)
    proc_file.set('sr0', sr0)
    proc_file.set('rg_sampling', rg_bw*over_fs)
    proc_file.set('rg_bw', rg_bw)
    proc_file.set('b_ati', b_ati)
    proc_file.set('b_xti', b_xti)
    proc_file.close()

    print('-----------------------------------------')
    print(time.strftime(
        "Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-----------------------------------------')

def ross_sar_focus(cfg_file, reconstruct_raw_output_file, output_file):

    ###################
    # INITIALIZATIONS #
    ###################

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR Processor for Rose-L: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('-------------------------------------------------------------------')

    # CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

    # PROCESSING
    az_weighting = cfg.processing.az_weighting
    doppler_bw = cfg.processing.doppler_bw
    plot_format = cfg.processing.plot_format
    plot_save = cfg.processing.plot_save
    plot_path = cfg.processing.plot_path
    plot_raw = cfg.processing.plot_raw
    plot_rcmc_dopp = cfg.processing.plot_rcmc_dopp
    plot_rcmc_time = cfg.processing.plot_rcmc_time
    plot_image_valid = cfg.processing.plot_image_valid

    # SAR
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    alt = cfg.sar.alt
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.sar.rg_bw
    over_fs = cfg.sar.over_fs

    # CALCULATE PARAMETERS
    l0 = const.c / f0
    if v_ground == 'auto':
        v_ground = geo.orbit_to_vel(
            alt, ground=True, inc=np.deg2rad(cfg.sar.inc_angle))
    rg_sampling = rg_bw * over_fs

    # RAW DATA
    raw_file = tpio.ReconstructedRawFile(reconstruct_raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    sr0 = raw_file.get('sr0')
    az0 = raw_file.get('az0')
    inc_angle = raw_file.get('inc_angle')
    # b_ati = raw_file.get('b_ati')
    # b_xti = raw_file.get('b_xti')
    raw_file.close()
    v_eff = estimate_effective_velocity(cfg, np.deg2rad(inc_angle))
    print("Effective focusing velocity: %.3f m/s" % v_eff)

    # OTHER INITIALIZATIONS
    # Create plots directory
    plot_path = os.path.dirname(output_file) + os.sep + plot_path
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    slc = []

    ########################
    # PROCESSING MAIN LOOP #
    ########################

    if plot_raw:
        plt.figure()
        plt.imshow(np.real(raw_data),
                    vmin=-np.max(np.abs(raw_data)),
                    vmax=np.max(np.abs(raw_data)), cmap='gray')
        plt.savefig(plot_path + os.sep + ('plot_raw_real.%s' % (plot_format)))
        plt.close()
        
    # Optimize matrix sizes
    az_size_orig, rg_size_orig = raw_data[0].shape
    optsize = utils.optimize_fftsize(raw_data[0].shape)
    optsize = [raw_data.shape[0], optsize[0], optsize[1]] # remove the hard coded number of 1
    data = np.zeros(optsize, dtype=complex)
    data[:, :raw_data[0].shape[0],
            :raw_data[0].shape[1]] = raw_data[:, :, :]

    az_size, rg_size = data.shape[1:]

    # RCMC Correction
    print('Applying RCMC correction... ')
    
    fr = np.fft.fftfreq(rg_size, 1/rg_sampling)
    fa = np.fft.fftfreq(az_size, 1/prf)
    ## Compensation of ANTENNA PATTERN
    ## FIXME this will not work for a long separation betwen Tx and Rx!!!
    sin_az = fa * l0 / (2 * v_eff)
    if hasattr(cfg.sar, 'ant_L'):
        ant_L = cfg.sar.ant_L
        beam_pattern = sinc_1tx_nrx(sin_az, ant_L, f0, 1, field=True)
    else:
        ant_l_tx = cfg.sar.ant_L_tx
        ant_l_rx = cfg.sar.ant_L_rx
        beam_pattern = (sinc_bp(sin_az, ant_l_tx, f0, field=True)
                        * sinc_bp(sin_az, ant_l_rx, f0, field=True))
    rcmc_fa = sr0 / np.sqrt(1 - (fa * (l0 / 2.) / v_eff)**2.) - sr0
    data = np.fft.fft(np.fft.fft(data, axis=-1), axis=-2)
    data = (data * np.exp(4j * np.pi * rcmc_fa.reshape((1, az_size, 1)) /
                            const.c * fr.reshape((1, 1, rg_size))))
    data = np.fft.ifft(data, axis=2)

    if plot_rcmc_dopp:
        plt.figure()
        plt.imshow(np.fft.fftshift(np.abs(data[0]), axes=0), vmax=np.max(np.abs(data)), cmap='gray',
                    origin='lower')
        plt.savefig(plot_path + os.sep + ('plot_rcmc_dopp.%s' % (plot_format)))

    if plot_rcmc_time:
        rcmc_time = np.fft.ifft(data[0], axis=0)[
            :az_size_orig, :rg_size_orig]
        rcmc_time_max = np.max(np.abs(rcmc_time))
        plt.figure()
        plt.imshow(np.real(rcmc_time), vmin=-rcmc_time_max, vmax=rcmc_time_max, cmap='gray',
                    origin='lower')
        plt.savefig(plot_path + os.sep + ('plot_rcmc_time_real.%s' % (plot_format)))

    # Azimuth compression
    print(
        'Applying azimuth compression... ')

    n_samp = 2 * (int(doppler_bw / (fa[1] - fa[0])) / 2)
    weighting = (az_weighting -
                    (1. - az_weighting) * np.cos(2 * np.pi * np.linspace(0, 1., int(n_samp))))
    # Compensate amplitude loss

    L_win = np.sum(np.abs(weighting)**2) / weighting.size
    weighting /= np.sqrt(L_win)
    if fa.size > n_samp:
        zeros = np.zeros(az_size)
        zeros[0:int(n_samp)] = weighting
        weighting = np.roll(zeros, int(-n_samp / 2))
    weighting = np.where(np.abs(beam_pattern) > 0, weighting/beam_pattern, 0)
    ph_ac = 4. * np.pi / l0 * sr0 * \
        (np.sqrt(1. - (fa * l0 / 2. / v_eff)**2.) - 1.)
    data = data * (np.exp(1j * ph_ac) * weighting).reshape((1, az_size, 1))

    data = np.fft.ifft(data, axis=1)

    print('Finishing... ')
    # Reduce to initial dimension
    data = data[:, :int(az_size_orig), :int(rg_size_orig)]

    # Removal of non valid samples
    n_val_az_2 = np.floor(
        doppler_bw / 2. / (2. * v_eff**2. / l0 / sr0) * prf / 2.) * 2.
    data = data[:, int(n_val_az_2):int(az_size_orig - n_val_az_2 - 1), :]
    if plot_image_valid:
        plt.figure()
        plt.imshow(np.abs(data[0]), origin='lower', vmin=0, vmax=np.max(np.abs(data)),
                    aspect=float(rg_size_orig) / float(az_size_orig),
                    cmap='gray')
        plt.xlabel("Range")
        plt.ylabel("Azimuth")
        plt.savefig(os.path.join(
            plot_path, ('plot_image_valid_.%s' % (plot_format))))

    slc.append(data)

    # Save processed data
    slc = np.array(slc, dtype=complex)
    print("Shape of SLC: " + str(slc.shape), flush=True)
    proc_file = tpio.ProcFile(output_file, 'w', slc.shape)
    proc_file.set('slc*', slc)
    proc_file.set('inc_angle', inc_angle)
    proc_file.set('f0', f0)
    proc_file.set('ant_L', ant_l_tx)
    proc_file.set('prf', prf)
    proc_file.set('v_ground', v_ground)
    proc_file.set('az0', az0+n_val_az_2*(v_ground/prf))
    proc_file.set('orbit_alt', alt)
    proc_file.set('sr0', sr0)
    proc_file.set('rg_sampling', rg_bw*over_fs)
    proc_file.set('rg_bw', rg_bw)
    proc_file.close()

    print('-----------------------------------------')
    print(time.strftime(
        "Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-----------------------------------------')

if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-r', '--raw_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    sar_focus(args.cfg_file, args.raw_file, args.output_file)
