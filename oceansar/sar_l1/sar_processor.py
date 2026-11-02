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
import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy import fft as sp_fft
from scipy import signal as scipy_signal

from drama.geo.geo_history import GeoHistory
from drama.performance.sar.sar_performance_common import calc_analysis_time
from oceansar.memory import memory_limited
from oceansar.utils import geometry as geo
from drama import utils
from oceansar import ocs_io as tpio
from oceansar import constants as const
from oceansar.radarsim.antenna_model import SarAntenna, resolve_acquisition
from oceansar.ocs_io import product_metadata as metadata
from oceansar.utils.plotting import finish_plot


def make_geohistory(cfg, inc_angle, fit_half_span=1.0):
    """Create a GeoHistory object covering the local processing interval."""
    alt = cfg.orbit.Horb
    t_analysis = max(
        2.0 * fit_half_span + 2.0,
        calc_analysis_time(
            alt, inc_angle, cfg.sar.f0, cfg.mode.prf, n_amb=1))
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


def mean_orbital_velocity(ghist):
    """Return the mean ECEF speed represented by a GeoHistory trajectory."""
    return np.mean(np.linalg.norm(ghist.v_ecef, axis=1))


def geohistory_azimuth_sine(ghist, look_angle, azimuth_position,
                            v_ground):
    """Return the raw-generator azimuth look sine on an image grid."""
    target_time = -np.asarray(azimuth_position)/v_ground
    return np.flip(
        ghist.t2u_spl(look_angle, np.flip(target_time)),
        axis=1,
    ).T.ravel()


def geohistory_doppler_sine(ghist, look_angle, fa, f0):
    """Return the raw-generator antenna look sine on a Doppler grid."""
    # Antenna angle obeys f_D = 2*v_orbit*u/lambda. The effective focusing
    # velocity describes range curvature and cannot replace orbital velocity
    # in this projection. Use the same GeoHistory angle mapping as the raw
    # generator, including its time-dependent orbital speed.
    order = np.argsort(fa)[::-1]
    _, sine_sorted, _, _, _ = ghist.Doppler2tuv(
        look_angle, np.asarray(fa)[order], f0)
    sine = np.empty(np.asarray(fa).shape, dtype=float)
    sine[order] = np.asarray(sine_sorted).ravel()
    return sine


def scene_coordinates(cfg):
    """Return the optimized raw-generator scene coordinates."""
    if hasattr(cfg.ocean, 'Nx') and hasattr(cfg.ocean, 'Ny'):
        nx, ny = int(cfg.ocean.Nx), int(cfg.ocean.Ny)
    else:
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


def geohistory_src_filter(ghist, look_angle, fa, fr, f0,
                          range_migration, azimuth_phase):
    """Return center-reference secondary range-compression phase."""
    rcmc_phase = (4*np.pi/const.c * range_migration[:, np.newaxis]
                  * fr[np.newaxis, :])
    src_phase = np.empty((fa.size, fr.size), dtype=np.float32)
    for rg_index, range_frequency in enumerate(fr):
        _, wideband_phase, _, _ = geohistory_reference_filter(
            ghist, look_angle, fa, f0 + range_frequency)
        src_phase[:, rg_index] = (
            wideband_phase - azimuth_phase - rcmc_phase[:, rg_index])
    return src_phase


def range_processing_grid(ghist, look_angle, raw_samples, sample_rate,
                           azimuth_sampling, f0):
    """Return fast range FFT length, minimum length and migration guard.

    raw_samples is the input range count; sample_rate, azimuth_sampling and
    f0 are header range sampling, actual azimuth sampling and carrier in Hz.
    ghist/look_angle specify the same reference geometry as RCMC and SRC.

    The combined RCMC/SRC phase is phi(f0+fr, fa)-phi(f0, fa). Its derivative
    with respect to fr is 4*pi*range_migration/c, giving a range-sample shift
    2*sample_rate*range_migration/c. Bound the nominal shift at both azimuth
    Nyquist edges and the lowest range-band carrier (largest migration).
    Keep ceil(max(abs(shift)))+1 zero samples after the input: a shift through
    the start then wraps into this discarded tail, not the retained image.
    The extra sample covers endpoint rounding. Fractional-shift/spectral tails
    are not finite-support kernels; this guard covers nominal group delay.

    Apply next_fast_len only after adding this guard. Fast-size rounding alone
    must never be relied on to supply range-migration padding.
    """
    lowest_carrier = f0-sample_rate/2
    if lowest_carrier <= 0:
        raise ValueError('Range sampling band must have positive carrier frequencies')
    migration, _, _, _ = geohistory_reference_filter(
        ghist, look_angle, np.array([-azimuth_sampling/2, azimuth_sampling/2]),
        lowest_carrier)
    shift = np.asarray(migration, dtype=np.float64)*2*sample_rate/const.c
    if not np.isfinite(shift).all():
        raise ValueError('Nonfinite range-migration guard')
    guard = int(np.ceil(np.max(np.abs(shift))))+1
    minimum = int(raw_samples)+guard
    return sp_fft.next_fast_len(minimum, real=False), minimum, guard


def point_target_azimuth_profile(data, expected_az, expected_rg,
                                 oversample=16, span=10, *, plot_save=True, plot_show=False):
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
                                   prf, plot_path, plot_format,
                                   oversample=16, span=10, *, plot_save=True, plot_show=False):
    """Plot retained target azimuth responses for one channel.

    expected_targets contains (name, azimuth index, range index) entries within
    data, whose axes are (polarization, azimuth, range). Cropping a ScanSAR
    image may retain fewer than the original nine scene targets.
    """
    if not expected_targets:
        return
    ncols = min(3, len(expected_targets))
    nrows = int(np.ceil(len(expected_targets)/ncols))
    def azimuth_spectrum(expected_az, expected_rg, cut_samples=128,
                         fft_oversample=8, phase_threshold_db=-12.0):
        """Return amplitude and detrended phase of a local target-cut FFT."""
        cut_samples = min(cut_samples, data.shape[1])
        az_center = int(np.round(expected_az))
        rg_center = int(np.round(expected_rg))
        az_slice = slice(max(0, az_center - 8),
                         min(data.shape[1], az_center + 9))
        rg_slice = slice(max(0, rg_center - 4),
                         min(data.shape[2], rg_center + 5))
        local_image = np.abs(data[0, az_slice, rg_slice])
        local_peak = np.unravel_index(
            np.argmax(local_image), local_image.shape)
        rg_peak = rg_slice.start + local_peak[1]

        cut_start = az_center - cut_samples//2
        cut_start = min(max(0, cut_start), data.shape[1] - cut_samples)
        profile = data[0, cut_start:cut_start + cut_samples, rg_peak]
        if profile.size < 2:
            raise ValueError(
                'Point-target azimuth cut is too short for an FFT')

        n_fft = profile.size*fft_oversample
        profile_padded = np.zeros(n_fft, dtype=complex)
        pad_start = (n_fft - profile.size)//2
        profile_padded[pad_start:pad_start + profile.size] = profile
        spectrum = sp_fft.fftshift(sp_fft.fft(
            sp_fft.ifftshift(profile_padded)))
        frequency = sp_fft.fftshift(sp_fft.fftfreq(n_fft, d=1/prf))
        amplitude = np.abs(spectrum)
        amplitude /= np.max(amplitude)
        amplitude_db = 20*np.log10(np.maximum(amplitude, 1e-6))

        # Ignore phase where crop leakage and numerical noise dominate.
        phase_support = amplitude_db >= phase_threshold_db
        center = n_fft//2
        left = center
        right = center
        while left > 0 and phase_support[left - 1]:
            left -= 1
        while right + 1 < n_fft and phase_support[right + 1]:
            right += 1
        support = slice(left, right + 1)
        phase_frequency = frequency[support]
        phase = np.unwrap(np.angle(spectrum[support]))
        if phase.size >= 2:
            linear_phase = np.polyval(np.polyfit(
                phase_frequency, phase, 1, w=amplitude[support]),
                phase_frequency)
            phase -= linear_phase

        return frequency, amplitude_db, phase_frequency, phase

    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 11*nrows/3),
                            sharex=True, sharey=True, squeeze=False)
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
    finish_plot(fig, os.path.join(
        plot_path, 'plot_point_target_azimuth_grid_%d.%s' %
        (channel, plot_format)), save=plot_save, show=plot_show)

    spectra = [azimuth_spectrum(expected_az, expected_rg)
               for _, expected_az, expected_rg in expected_targets]

    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 11*nrows/3),
                            sharex=True, sharey=True, squeeze=False)
    for ax, (target, spectrum) in zip(
            axs.ravel(), zip(expected_targets, spectra)):
        frequency, amplitude_db, _, _ = spectrum
        ax.plot(frequency, amplitude_db)
        ax.set_title(target[0].replace('_', ' / '))
        ax.set_xlim(-prf/2, prf/2)
        ax.set_ylim(-40, 1)
        ax.grid(True)
    for ax in axs[-1, :]:
        ax.set_xlabel('Doppler frequency [Hz]')
    for ax in axs[:, 0]:
        ax.set_ylabel('Normalized amplitude [dB]')
    fig.suptitle('Point-target azimuth spectra, channel %d' % channel)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finish_plot(fig, os.path.join(
        plot_path, 'plot_point_target_azimuth_spectrum_amplitude_%d.%s' %
        (channel, plot_format)), save=plot_save, show=plot_show)

    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 11*nrows/3),
                            sharex=True, sharey=True, squeeze=False)
    for ax, (target, spectrum) in zip(
            axs.ravel(), zip(expected_targets, spectra)):
        _, _, phase_frequency, phase = spectrum
        ax.plot(phase_frequency, phase)
        ax.set_title(target[0].replace('_', ' / '))
        ax.set_xlim(-prf/2, prf/2)
        ax.grid(True)
    for ax in axs[-1, :]:
        ax.set_xlabel('Doppler frequency [Hz]')
    for ax in axs[:, 0]:
        ax.set_ylabel('Residual phase [rad]')
    fig.suptitle(
        'Point-target azimuth spectral phase, channel %d\n'
        '(constant and linear phase removed)' % channel)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    finish_plot(fig, os.path.join(
        plot_path, 'plot_point_target_azimuth_spectrum_phase_%d.%s' %
        (channel, plot_format)), save=plot_save, show=plot_show)


def check_reference_range_history(cfg, ghist, inc_angle, raw_sr0, az0,
                                  prf, v_ground, f0, sr_pt, az_size,
                                  v_eff, plot_path=None, plot_format='png',
                                  plot_save=False, plot_show=False):
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

    fa = sp_fft.fftfreq(az_size, 1/prf)
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

    if (plot_save or plot_show) and plot_path is not None:
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
        finish_plot(fig, os.path.join(
            plot_path, 'plot_range_history_check.%s' % plot_format), save=plot_save, show=plot_show)

def azimuth_common_band_filter(fa, doppler_bw, az_weighting):
    """Return a zero-centred, unit mean in-band power amplitude weighting.

    fa is the actual FFT frequency grid in Hz; doppler_bw is the full retained
    bandwidth in Hz and az_weighting is the raised-cosine constant coefficient
    (1 for rectangular, .5 for Hann). Evaluate an even function of frequency
    directly: rolling an even-length window centres it at minus half a bin,
    introducing a deterministic Doppler offset. This also centres a full-band
    taper correctly, including the unpaired even-FFT Nyquist sample.
    """
    fa = np.asarray(fa, dtype=float)
    if (fa.ndim != 1 or fa.size < 3 or not np.isfinite(fa).all()
            or not np.isfinite(doppler_bw) or doppler_bw <= 0
            or not np.isfinite(az_weighting)):
        raise ValueError('Invalid azimuth frequency grid, bandwidth or weighting')
    spacing = np.min(np.diff(np.sort(fa)))
    if spacing <= 0:
        raise ValueError('Azimuth frequency grid must contain distinct bins')
    bandwidth = min(float(doppler_bw), fa.size * spacing)
    support = np.abs(fa) <= bandwidth/2
    if np.count_nonzero(support) < 3:
        raise ValueError("doppler_bw is too narrow for the azimuth grid")
    weighting = np.zeros(fa.size)
    weighting[support] = (az_weighting + (1.0 - az_weighting)
                          * np.cos(2*np.pi*fa[support]/bandwidth))
    window_power = np.mean(weighting[support]**2)
    if window_power <= 0:
        raise ValueError('Azimuth weighting has zero power')
    weighting /= np.sqrt(window_power)
    return weighting


def compensate_azimuth_antenna(data, beam_pattern):
    """Compensate the antenna response in the original Doppler domain."""
    compensation = np.zeros_like(beam_pattern, dtype=complex)
    valid = np.abs(beam_pattern) > 0
    compensation[valid] = 1.0 / beam_pattern[valid]
    return data * compensation[np.newaxis, :, np.newaxis]


def processing_antenna_pattern(cfg, beam_pattern):
    """Return the complex field actually divided out during focusing.

    Parameters
    ----------
    cfg : ConfigFile
        processing.antenna_equalization (default True) enables division.
        antenna_min_relative_field (default 0.001) suppresses bins below this
        fraction of the largest field on the processing grid, avoiding noise
        amplification at array nulls. Zero requests exact nonzero inversion.
    beam_pattern : 1D complex array
        Acquired Tx times Rx field on the processing Doppler grid.

    Returns
    -------
    ndarray
        Effective complex field; masked bins are zero and compensation sets
        those data bins to zero. Disabled equalization returns unity. Save
        this same array in the DC response metadata, including its null mask.
    """
    if not getattr(cfg.processing, 'antenna_equalization', True):
        return np.ones_like(beam_pattern)
    minimum = float(getattr(cfg.processing, 'antenna_min_relative_field', .001))
    if not np.isfinite(minimum) or not 0 <= minimum < 1:
        raise ValueError('processing.antenna_min_relative_field must be in [0, 1)')
    field = np.asarray(beam_pattern)
    if not np.isfinite(field).all():
        raise ValueError('Antenna field must be finite')
    return np.where(abs(field) >= minimum*np.max(abs(field)), field, 0)


def azimuth_filter_samples(sample_rate, doppler_rate, bandwidth=None):
    """Return nominal azimuth focusing-filter support in time samples.

    sample_rate is actual azimuth sampling in Hz (header az_sampling).
    doppler_rate contains the nonzero local Doppler rates in Hz/s across the
    processed range. Use the smallest magnitude for the longest filter.
    bandwidth is the focused bandwidth in Hz; None selects the full sampled
    band, as required by ScanSAR's focus-before-deramp processing.

    A chirp sweeps bandwidth B in T = B / min(abs(Ka)) seconds. Its matched
    filter therefore spans Nh = ceil(sample_rate * B / min(abs(Ka))) samples.
    ScanSAR uses B = sample_rate, giving ceil(sample_rate**2 / abs(Ka));
    stripmap applies the common-band window with the matched filter and can
    use B = min(processed_bandwidth, sample_rate).

    This is the nominal chirp support, not an explicit truncation of the
    frequency-domain filter. Spectral-cutoff tails and the later ScanSAR
    low-pass response can extend beyond it; they are not included in Nh.
    """
    if (isinstance(sample_rate, (bool, np.bool_)) or not np.isscalar(sample_rate)
            or not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError('sample_rate must be positive and finite')
    rates = np.asarray(doppler_rate, dtype=np.float64)
    if rates.size == 0 or not np.isfinite(rates).all() or np.any(rates == 0):
        raise ValueError('Doppler rates must be finite and nonzero')
    if bandwidth is None:
        bandwidth = sample_rate
    if (isinstance(bandwidth, (bool, np.bool_)) or not np.isscalar(bandwidth)
            or not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('bandwidth must be positive and finite')
    return max(1, int(np.ceil(sample_rate * min(bandwidth, sample_rate)
                              / np.min(np.abs(rates)))))


def scansar_azimuth_grid(scene_length, image_length, sample_rate, v_ground,
                         raw_samples, *, doppler_rate, full_image=False):
    """Return ScanSAR output count, minimum FFT count and crop offset.

    scene_length and image_length are along-track lengths in metres; None for
    image_length preserves the scene extent for unannotated acquisitions.
    sample_rate is the actual input sampling in Hz, not the nominal PRF;
    v_ground is in m/s and raw_samples is the acquired burst sample count.
    doppler_rate contains the range-dependent rates in Hz/s.
    full_image requests the nominal image even when the scene is shorter.

    The crop offset is relative to the first acquired pulse. Centre on the raw
    burst, resolving a half-sample tie towards the later sample. It must not
    depend on FFT padding parity. Ceil rounds lengths up by less than a pixel.

    For raw length Nr and nominal focusing support Nh, linear convolution
    needs Nr + Nh - 1 samples. The minimum FFT grid is therefore
    max(Nr + Nh - 1, output_samples). Round up with scipy.fft.next_fast_len
    afterwards. The simulated scene length does not add a convolution guard:
    contributions from the entire scene are already present in the raw burst,
    including physical azimuth ambiguities. See azimuth_filter_samples for
    the chirp-support approximation and its spectral-tail limitations.
    """
    for name, value in (('scene_length', scene_length),
                        ('sample_rate', sample_rate), ('v_ground', v_ground)):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be positive and finite')
    if image_length is None:
        image_length = scene_length
    if (isinstance(image_length, (bool, np.bool_)) or not np.isscalar(image_length)
            or not np.isfinite(image_length) or image_length <= 0):
        raise ValueError('image_length must be positive and finite, in metres')
    if (isinstance(raw_samples, (bool, np.bool_)) or not np.isfinite(raw_samples)
            or int(raw_samples) != raw_samples or raw_samples < 1):
        raise ValueError('raw_samples must be a positive integer')
    if not isinstance(full_image, (bool, np.bool_)):
        raise ValueError('processing.scansar_full_image must be boolean')
    length = image_length if full_image else min(scene_length, image_length)
    output_samples = max(1, int(np.ceil(length * sample_rate / v_ground)))
    filter_samples = azimuth_filter_samples(sample_rate, doppler_rate)
    minimum_fft = max(int(raw_samples) + filter_samples - 1, output_samples)
    crop_offset = (int(raw_samples) - output_samples + 1) // 2
    return output_samples, minimum_fft, crop_offset


def processing_azimuth_fft_size(cfg, minimum, default):
    """Return a validated optional exact FFT length for processing diagnostics.

    processing.azimuth_fft_size=None preserves the optimized default. An
    explicit integer must cover the required processing support and guards.
    This changes zero padding/frequency sampling, never the acquisition PRF.
    Actual frequency weights are saved in the SLC for DC response correction.
    """
    value=getattr(cfg.processing,'azimuth_fft_size',None)
    if value is None or str(value).lower()=='none':return int(default)
    if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or int(value)!=value or value<minimum:
        raise ValueError('processing.azimuth_fft_size must be an integer covering processing support and zero guards')
    return int(value)


def scansar_deramp_phase(az_size, slant_range, az0, prf,
                         v_ground, v_eff, wavelength):
    """Compute the geometric ScanSAR deramp phase on the focused image grid.

    Parameters
    ----------
    az_size : int
        Number of azimuth samples.
    slant_range : 1D array
        Closest-approach range per range bin in metres.
    az0 : float
        Along-track position of the first image row in metres.
    prf : float
        Azimuth sampling rate in Hz.
    v_ground : float
        Ground-track speed in m/s, converting az0 to the first sample time.
    v_eff : float
        Effective velocity in m/s for the hyperbolic slant-range model.
    wavelength : float
        Radar wavelength in metres.

    Returns
    -------
    numpy.ndarray
        Phase in radians with shape (az_size, number_of_range_bins). Multiply
        image samples by exp(-1j * phase) to deramp. The phase is
        4*pi/wavelength * (sqrt(R0**2 + (v_eff*t)**2) - R0), evaluated using
        a rationalized range difference for accuracy near closest approach.
    """
    azimuth_time = az0/v_ground + np.arange(az_size)/prf
    slant_range = np.asarray(slant_range)[np.newaxis, :]
    along_track_squared = (v_eff * azimuth_time[:, np.newaxis])**2
    instantaneous_range = np.sqrt(slant_range**2 + along_track_squared)
    # Within the effective-velocity hyperbolic range model, the phase removed
    # by exp(-1j*phase) is 4*pi/lambda * (R(t)-R(0)). It is the integrated
    # geometric phase, not its time derivative (a Doppler-related quantity).
    # Near t=0 this reduces to the former 2*pi*v_eff**2*t**2/(lambda*R(0)).
    # Rationalize the range difference to avoid cancellation near t=0.
    return (4*np.pi / wavelength * along_track_squared
            / (instantaneous_range + slant_range))


def stripmap_azimuth_focus(data, ph_ac, common_band_filter, *, workers=None):
    """Return stripmap SLC samples with the same shape as complex data.

    data has axes (batch, azimuth-Doppler, range); sar_focus batches the
    polarizations of one channel. ph_ac is matched-filter phase in radians,
    shaped (azimuth,) or (azimuth, range). common_band_filter contains the
    one-dimensional amplitude weights in azimuth FFT order. workers is the
    SciPy FFT thread count, or None for its current default. The returned
    azimuth axis is in the spatial domain; no image cropping is done here.
    """
    az_size = data.shape[1]

    if ph_ac.ndim == 1:
        azimuth_filter = (
            np.exp(1j * ph_ac) * common_band_filter
        ).reshape((1, az_size, 1))
    else:
        azimuth_filter = (
            np.exp(1j * ph_ac)[np.newaxis, :, :]
            * common_band_filter.reshape((1, az_size, 1))
        )

    return sp_fft.ifft(data * azimuth_filter, axis=1, workers=workers)


def scansar_azimuth_focus(
    data,
    ph_ac,
    deramp_phase,
    common_band_filter=None,
    *,
    reramp=True,
    workers=None,
):
    """Focus ScanSAR data using deramp-filter-reramp processing.

    The input data are assumed to have already undergone any desired
    antenna-pattern compensation.

    Parameters
    ----------
    data : ndarray
        RCMC-corrected data in azimuth-Doppler/range coordinates.

        Expected shape:
        ``(batch, azimuth, range)``. sar_focus batches the polarizations
        of one channel.

    ph_ac : ndarray
        Azimuth matched-filter phase in the original Doppler domain.

        Supported shapes:
        ``(azimuth,)`` or ``(azimuth, range)``.

    deramp_phase : ndarray
        Phase removed in the focused azimuth/range domain before
        transforming to the common baseband Doppler domain.

        Supported shapes:
        ``(azimuth,)`` or ``(azimuth, range)``.

    common_band_filter : ndarray or None, optional
        Filter applied in the deramped baseband Doppler domain.

        Supported shapes:
        ``(azimuth,)`` or ``(azimuth, range)``.

        If None, no baseband filtering is applied. This is useful while
        testing the deramp-reramp operation, although in normal ScanSAR
        processing a common-band selection will generally be required.

    reramp : bool, optional
        If True, restore the removed azimuth phase before returning.
        If False, return the focused data in deramped coordinates.

    workers : int or None, optional
        SciPy FFT worker count. None uses SciPy's current default; sar_focus
        supplies sim.nworkers to each transform.

    Returns
    -------
    ndarray
        Focused ScanSAR data with the same shape as ``data``.
    """
    data = np.asarray(data)
    ph_ac = np.asarray(ph_ac)
    deramp_phase = np.asarray(deramp_phase)

    if data.ndim != 3:
        raise ValueError(
            "data must have shape (batch, azimuth, range)"
        )

    az_size = data.shape[1]
    range_size = data.shape[2]

    def _broadcast_azimuth_quantity(quantity, name):
        """Convert an azimuth quantity to broadcastable 3-D form."""
        if quantity.ndim == 1:
            if quantity.shape[0] != az_size:
                raise ValueError(
                    f"{name} has azimuth size {quantity.shape[0]}, "
                    f"expected {az_size}"
                )
            return quantity[None, :, None]

        if quantity.ndim == 2:
            if quantity.shape != (az_size, range_size):
                raise ValueError(
                    f"{name} has shape {quantity.shape}, expected "
                    f"({az_size}, {range_size})"
                )
            return quantity[None, :, :]

        raise ValueError(
            f"{name} must have shape (azimuth,) "
            f"or (azimuth, range)"
        )

    ph_ac = _broadcast_azimuth_quantity(ph_ac, "ph_ac")
    deramp_phase = _broadcast_azimuth_quantity(
        deramp_phase,
        "deramp_phase",
    )

    # 1. Apply the azimuth matched filter in the original Doppler domain.
    azimuth_filter = np.exp(1j * ph_ac)

    # 2. Transform to focused azimuth/range coordinates.
    focused = sp_fft.ifft(data * azimuth_filter, axis=1, workers=workers)

    # 3. Remove the azimuth-dependent phase modulation.
    deramp = np.exp(-1j * deramp_phase)
    deramped = focused * deramp

    # 4. Transform to the common baseband Doppler domain.
    baseband_spectrum = sp_fft.fft(deramped, axis=1, workers=workers)

    # 5. Select the desired common Doppler band.
    if common_band_filter is not None:
        common_band_filter = _broadcast_azimuth_quantity(
            np.asarray(common_band_filter),
            "common_band_filter",
        )
        baseband_spectrum *= common_band_filter

    # 6. Return to deramped azimuth/range coordinates.
    result = sp_fft.ifft(baseband_spectrum, axis=1, workers=workers)

    # 7. Optionally restore the removed phase.
    if reramp:
        result *= np.conj(deramp)

    return result


@memory_limited
def sar_focus(cfg_file, raw_output_file, output_file, *, plot_show=False):

    ###################
    # INITIALIZATIONS #
    ###################

    """Focus complex SAR raw data into a complex SLC file.

    Parameters
    ----------
    cfg_file : str or path-like
        Configuration supplying processing controls, including Doppler
        bandwidth, azimuth weighting, diagnostic plots and ScanSAR reramping.
    raw_output_file : str or path-like
        System-processed RawFile containing calibrated complex samples with
        axes (polarization, channel, azimuth, range), geometry and baselines.
        Normally produced by sar_system_errors. Acquisition and geometry
        headers are authoritative. az_sampling (Hz) specifies actual sample
        timing after any upsampling; prf remains the acquisition pulse rate.
        Single reconstructed channels must use an explicit length-one channel
        axis and corresponding num_ch/baselines in this interface.
    output_file : str or path-like
        Destination ProcFile. slc_r/slc_i store the focused complex image with
        axes (channel, polarization, azimuth, range). Geometry metadata include
        az0 and sr0 in metres, sampling rates and carrier frequency in Hz,
        velocities in m/s and incidence angle in degrees.
    plot_show : bool, optional
        Display enabled diagnostic figures, including when also saving them.
        Unsaved enabled figures are displayed automatically. Figures close
        after saving/displaying. Default False.

    Returns
    -------
    None
        Write the SLC and configured diagnostics to disk.

    Notes
    -----
    This function performs focusing only; it does not apply cal_factor, add
    receiver noise or delete its input. The caller orchestrates those steps.
    ScanSAR processing.reramp selects whether to restore the azimuth phase
    ramp after deramped focusing; the stored phase reference matters for DC.
    ScanSAR SLCs also store burst timing and response-model power weights
    for downstream Doppler sensitivity correction.
    The acquired mode image_length (metres) caps the ScanSAR output extent.
    By default shorter scenes are retained at their own length; setting
    processing.scansar_full_image=True retains the full nominal image instead.
    FFT zero guards use the nominal focusing-filter duration B/abs(Ka),
    independently of the simulated scene extent. scipy.fft.next_fast_len
    selects efficient mixed-factor sizes; sim.nworkers selects FFT threads.
    sim.memory_limit_gib optionally isolates this stage and monitors measured
    caller+worker memory, stopping near the configured GiB budget.
    """
    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR Processor: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('-------------------------------------------------------------------')

    # CONFIGURATION FILE
    from oceansar.radarsim.sar_system_errors import validate_raw_paths
    validate_raw_paths(raw_output_file, output_file)
    cfg = resolve_acquisition(tpio.ConfigFile(cfg_file))
    workers = getattr(cfg.sim, 'nworkers', 1)
    if (isinstance(workers, (bool, np.bool_)) or not np.isscalar(workers)
            or not np.isfinite(workers) or int(workers) != workers or workers < 1):
        raise ValueError('sim.nworkers must be a positive integer')
    workers = int(workers)

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
    reramp = getattr(cfg.processing, 'reramp', True)
    add_point_target = (cfg.sim.add_point_target
                        if hasattr(cfg.sim, 'add_point_target') else False)

    # RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    try:
        cfg = metadata.focusing_configuration(cfg, raw_file.__file__)
        antennas = SarAntenna.from_product(copy.deepcopy(cfg), raw_file.__file__)
        raw_data = raw_file.get('raw_data*')
        if raw_data.ndim != 4:
            raise ValueError('Expected raw dimensions (polarization, channel, azimuth, range)')
        num_ch = raw_data.shape[1]
        if metadata.scalar(raw_file.__file__, 'num_ch', positive=True) != num_ch:
            raise ValueError('Raw num_ch header does not match the channel dimension')
        labels = metadata.polarizations(raw_file.__file__, raw_data.shape[0], cfg)
        # Use actual sample timing, never acquisition PRF times an assumed channel
        # count. Reconstruction/upsampling must explicitly update az_sampling.
        prf = metadata.azimuth_sampling(raw_file.__file__)
        acquisition_prf = metadata.scalar(raw_file.__file__, 'prf', positive=True)
        f0 = metadata.scalar(raw_file.__file__, 'f0', positive=True)
        alt = metadata.scalar(raw_file.__file__, 'orbit_alt', positive=True)
        rg_sampling = metadata.scalar(raw_file.__file__, 'rg_sampling', positive=True)
        rg_bw = metadata.scalar(raw_file.__file__, 'rg_bw', positive=True)
        operation_mode = metadata.acquisition_mode(raw_file.__file__, cfg)
        if operation_mode not in ('stripmap', 'scansar'):
            raise ValueError('Unsupported SAR operating mode: %s' % operation_mode)
        l0 = const.c / f0
        sr0 = metadata.scalar(raw_file.__file__, 'sr0', positive=True)
        az0 = metadata.scalar(raw_file.__file__, 'az0')
        v_ground = metadata.scalar(raw_file.__file__, 'v_ground', positive=True)
        inc_angle = metadata.scalar(raw_file.__file__, 'inc_angle')
        b_ati = raw_file.get('b_ati')
        b_xti = raw_file.get('b_xti')
        for name, baseline in (('b_ati', b_ati), ('b_xti', b_xti)):
            if (np.ma.is_masked(baseline) or np.shape(baseline) != (num_ch,)
                    or not np.isfinite(baseline).all()):
                raise ValueError('Invalid raw baseline header: ' + name)
        sr_pt = None
        if 'sr_pt' in raw_file.__file__.variables:
            sr_pt = raw_file.get('sr_pt')
    finally:
        raw_file.close()
    inc_angle_rad = np.asarray(np.deg2rad(inc_angle)).item()
    ghist = make_geohistory(cfg, inc_angle_rad)
    v_eff = estimate_effective_velocity(cfg, inc_angle_rad, ghist=ghist)
    v_orbit = mean_orbital_velocity(ghist)
    _, look_focus, _ = reference_point_geometry(cfg, inc_angle_rad)
    point_targets = point_target_geometry(cfg, inc_angle_rad)
    print("Effective focusing velocity: %.3f m/s" % v_eff)
    print("Mean GeoHistory orbital velocity: %.3f m/s" % v_orbit)
    print("Ground velocity read from raw data: %.3f m/s" % v_ground)
    print("Range-dependent azimuth compression: %s" %
          range_dependent_azimuth)
    print("SAR operating mode: %s" % operation_mode)

    # OTHER INITIALIZATIONS
    # Create plots directory
    plot_path = os.path.join(os.path.dirname(output_file), plot_path)
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    slc = []
    filter_cache = {}
    output_az0 = None

    ########################
    # PROCESSING MAIN LOOP #
    ########################
    for ch in np.arange(num_ch):

        # Build the azimuth processing grid. ScanSAR acquisition duration and
        # image extent are independent. Retain the mode-defined image, while
        # allowing a larger simulated scene to contribute physical ambiguities.
        az_size_orig, rg_size_orig = raw_data[0, ch].shape
        rg_fft_size, minimum_rg_size, range_guard = range_processing_grid(
            ghist, look_focus, rg_size_orig, rg_sampling, prf, f0)
        # Use the farthest processed range for the smallest |Ka| and longest
        # filter. Keep range/phase geometry in float64. In reconstructed data
        # prf here is header az_sampling, not the nominal acquisition PRF.
        far_range = sr0 + (rg_fft_size-1)*const.c/(2*rg_sampling)
        minimum_doppler_rate = 2*v_eff**2/(l0*far_range)
        focusing_bandwidth = prf if operation_mode == 'scansar' else min(prf, doppler_bw)
        filter_samples = azimuth_filter_samples(
            prf, minimum_doppler_rate, focusing_bandwidth)
        if ch == 0 and sr_pt is not None:
            check_reference_range_history(
                cfg, ghist, inc_angle_rad, sr0, az0, prf, v_ground, f0,
                sr_pt, az_size_orig, v_eff, plot_path=plot_path,
                plot_format=plot_format, plot_save=plot_save, plot_show=plot_show)
        if operation_mode == 'scansar':
            output_az_size, minimum_az_size, output_offset = scansar_azimuth_grid(
                cfg.ocean.Ly, getattr(cfg.mode, 'image_length', None), prf,
                v_ground, az_size_orig, doppler_rate=minimum_doppler_rate,
                full_image=getattr(
                    cfg.processing, 'scansar_full_image', False))
        else:
            output_az_size = None
            minimum_az_size = az_size_orig + filter_samples - 1
        # Complex transforms: let SciPy choose supported small prime factors,
        # rather than rounding to the next power of two. An explicit diagnostic
        # azimuth_fft_size remains allowed, but must include the support above.
        optsize = [raw_data.shape[0],
                   sp_fft.next_fast_len(minimum_az_size, real=False), rg_fft_size]
        optsize[1] = processing_azimuth_fft_size(cfg, minimum_az_size, optsize[1])
        data = np.zeros(optsize, dtype=complex)
        az_size, rg_size = data.shape[1:]
        if operation_mode == 'scansar':
            az_insert_start = (az_size - az_size_orig) // 2
        else:
            az_insert_start = 0
        az_insert_stop = az_insert_start + az_size_orig
        data[:, az_insert_start:az_insert_stop,
             :rg_size_orig] = raw_data[:, ch, :, :]
        processing_az0 = (
            az0 - az_insert_start * v_ground / prf)
        if ch == 0 and operation_mode == 'scansar':
            print(
                "ScanSAR azimuth grid: %d burst samples, "
                "%d nominal filter samples (min |Ka| %.3f Hz/s), "
                "%d FFT samples, %d output samples; %d FFT workers"
                % (az_size_orig, filter_samples, minimum_doppler_rate,
                   az_size, output_az_size, workers))

        # RCMC Correction
        print('Applying RCMC correction... [Channel %d/%d]' % (ch + 1, num_ch))
        fr = sp_fft.fftfreq(rg_size, 1/rg_sampling)
    
        fa = sp_fft.fftfreq(az_size, 1/prf)
        slant_range = (sr0 + np.arange(rg_size)
                       * const.c/(2*rg_sampling))
        # Antenna response in the original azimuth-Doppler domain.
        ## FIXME this will not work for a long separation betwen Tx and Rx!!!
        sin_az = geohistory_doppler_sine(ghist, look_focus, fa, f0)
        beam_pattern = antennas.field(sin_az)
        beam_pattern = processing_antenna_pattern(cfg, beam_pattern)
        #fa[az_size/2:] = fa[az_size/2:] - prf
        filter_key = (az_size, rg_size)
        if filter_key not in filter_cache:
            rcmc_fa, ph_ac_reference, _, _ = (
                geohistory_reference_filter(ghist, look_focus, fa, f0))
            ph_src = geohistory_src_filter(
                ghist, look_focus, fa, fr, f0,
                rcmc_fa, ph_ac_reference)
            if range_dependent_azimuth:
                ph_ac = geohistory_range_azimuth_filter(
                    ghist, slant_range, fa, f0)
            else:
                ph_ac = ph_ac_reference
            common_band_filter = azimuth_common_band_filter(
                fa, doppler_bw, az_weighting)
            if operation_mode == 'scansar':
                deramp_phase = scansar_deramp_phase(
                    az_size, slant_range, processing_az0, prf,
                    v_ground, v_eff, l0)
            else:
                deramp_phase = None
            filter_cache[filter_key] = (
                rcmc_fa, ph_src, ph_ac,
                common_band_filter, deramp_phase)
        (rcmc_fa, ph_src, ph_ac,
         common_band_filter, deramp_phase) = filter_cache[filter_key]
        #rcmc_fa[:]=0
        data = sp_fft.fftn(data, axes=(-1, -2), workers=workers)

        range_doppler_phase = (
            4*np.pi/const.c * rcmc_fa[:, np.newaxis]
            * fr[np.newaxis, :] + ph_src)
        data = data * np.exp(1j*range_doppler_phase[np.newaxis, :, :])
        data = sp_fft.ifft(data, axis=2, workers=workers)

        if plot_rcmc_dopp:
            plt.figure()
            plt.imshow(sp_fft.fftshift(np.abs(data[0]), axes=0), vmax=np.max(np.abs(data)), cmap='gray',
                       origin='lower')
            finish_plot(plt.gcf(), os.path.join(plot_path, 'plot_rcmc_dopp_%d.%s' % (ch, plot_format)),
                        save=plot_save, show=plot_show)

        if plot_rcmc_time:
            rcmc_time = sp_fft.ifft(data[0], axis=0, workers=workers)[
                az_insert_start:az_insert_stop, :rg_size_orig]
            rcmc_time_max = np.max(np.abs(rcmc_time))
            plt.figure()
            plt.imshow(np.real(rcmc_time), vmin=-rcmc_time_max, vmax=rcmc_time_max, cmap='gray',
                       origin='lower')
            finish_plot(plt.gcf(), os.path.join(plot_path, 'plot_rcmc_time_real_%d.%s' % (ch, plot_format)),
                        save=plot_save, show=plot_show)

        # Azimuth compression
        print(
            'Applying azimuth compression... [Channel %d/%d]' % (ch + 1, num_ch))
        data = compensate_azimuth_antenna(data, beam_pattern)
        if operation_mode == 'scansar':
            data = scansar_azimuth_focus(
                data,
                ph_ac,
                deramp_phase,
                common_band_filter=common_band_filter,
                reramp=reramp,
                workers=workers,
            )
        else:
            data = stripmap_azimuth_focus(
                data, ph_ac, common_band_filter, workers=workers)

        print('Finishing... [Channel %d/%d]' % (ch + 1, num_ch))
        data = data[:, :, :rg_size_orig]

        # Stripmap requires an aperture-length edge crop. ScanSAR retains the
        # requested image portion, with coordinates independent of FFT length.
        if operation_mode == 'scansar':
            n_val_az_2 = 0
            output_start = az_insert_start + output_offset
            output_stop = output_start + output_az_size
            data = data[:, output_start:output_stop, :]
            output_az0 = (
                processing_az0 + output_start * v_ground / prf)
        else:
            n_val_az_2 = np.floor(
                doppler_bw / 2.
                / (2. * v_eff**2. / l0 / sr0)
                * prf / 2.) * 2.
            data = data[
                :,
                int(n_val_az_2):int(az_size_orig - n_val_az_2 - 1),
                :,
            ]
            output_az0 = az0 + n_val_az_2 * v_ground / prf
        if add_point_target and (plot_save or plot_show):
            expected_targets = []
            for target_name, target_look, target_y in point_targets:
                target_sr = ghist.sr_spl(target_look, 0.0).item()
                expected_az = (
                    (target_y - output_az0)*prf/v_ground)
                expected_rg = (target_sr - sr0)*2*rg_sampling/const.c
                if (0 <= expected_az < data.shape[1]
                        and 0 <= expected_rg < data.shape[2]):
                    expected_targets.append(
                        (target_name, expected_az, expected_rg))
                else:
                    print('Skipping point target outside retained SLC: ' + target_name)
            if expected_targets:
                plot_point_target_azimuth_grid(
                    data, expected_targets, ch, prf, plot_path, plot_format,
                    plot_save=plot_save, plot_show=plot_show)
        if plot_image_valid:
            plt.figure()
            range_step = const.c/(2*rg_sampling)
            azimuth_step = v_ground/prf
            plt.imshow(np.abs(data[0]), origin='lower', vmin=0, vmax=np.max(np.abs(data)),
                       extent=[-.5*range_step, (rg_size_orig-.5)*range_step,
                               output_az0-.5*azimuth_step,
                               output_az0+(data.shape[1]-.5)*azimuth_step],
                       aspect='equal',
                       cmap='gray')
            plt.xlabel("Slant range offset [m]")
            plt.ylabel("Azimuth [m]")
            finish_plot(plt.gcf(), os.path.join(
                plot_path, 'plot_image_valid_%d.%s' % (ch, plot_format)),
                save=plot_save, show=plot_show)

        slc.append(data)

    # Save processed data
    slc = np.array(slc, dtype=complex)
    print("Shape of SLC: " + str(slc.shape), flush=True)
    output_azimuth = (
        output_az0 + np.arange(slc.shape[2]) * v_ground/prf)
    output_sin_az = geohistory_azimuth_sine(
        ghist, look_focus, output_azimuth, v_ground)
    proc_file = tpio.ProcFile(output_file, 'w', slc.shape)
    proc_file.set('slc*', slc)
    proc_file.set('inc_angle', inc_angle)
    proc_file.set('f0', f0)
    proc_file.set('num_ch', num_ch)
    proc_file.set('ant_L', antennas.tx.La)
    antennas.save(proc_file.__file__)
    metadata.write_sampling(proc_file.__file__, prf, acquisition_prf)
    metadata.write_geometry(proc_file.__file__, cfg, labels)
    proc_file.__file__.processed_doppler_bandwidth_hz = float(min(doppler_bw, prf))
    proc_file.set('v_ground', v_ground)
    proc_file.set('v_orbit', v_orbit)
    proc_file.set('az0', output_az0)
    proc_file.set('reramp', reramp)
    proc_file.set('sin_az', output_sin_az)
    proc_file.set('orbit_alt', alt)
    proc_file.set('sr0', sr0)
    proc_file.set('rg_sampling', rg_sampling)
    proc_file.set('rg_bw', rg_bw)
    proc_file.set('b_ati', b_ati)
    proc_file.set('b_xti', b_xti)
    proc_file.__file__.fft_workers = workers
    proc_file.__file__.azimuth_focusing_filter_samples = int(filter_samples)
    proc_file.__file__.azimuth_doppler_rate_min_hz_per_s = float(minimum_doppler_rate)
    proc_file.__file__.azimuth_focusing_bandwidth_hz = float(focusing_bandwidth)
    proc_file.__file__.range_fft_size = int(rg_size)
    proc_file.__file__.range_fft_minimum_size = int(minimum_rg_size)
    proc_file.__file__.range_migration_guard_samples = int(range_guard)
    if operation_mode == 'scansar':
        proc_file.__file__.scansar_full_image = int(getattr(
            cfg.processing, 'scansar_full_image', False))
        proc_file.__file__.output_azimuth_length_m = slc.shape[2]*v_ground/prf
        proc_file.__file__.azimuth_fft_size = int(az_size)
        proc_file.__file__.azimuth_fft_minimum_size = int(minimum_az_size)
        proc_file.__file__.azimuth_zero_padding_samples = np.array(
            [az_insert_start, az_size-az_insert_stop], dtype=np.int64)
        from oceansar.radarsim.scansar_dc import write_scansar_dc_metadata
        write_scansar_dc_metadata(cfg, proc_file, antennas=antennas, ghist=ghist, look=look_focus,
            frequencies=fa, beam_pattern=beam_pattern, common_band_filter=common_band_filter,
            v_eff=v_eff, raw_az0=float(az0), raw_az_size=raw_data.shape[-2],
            fs_az=float(prf), v_ground=float(v_ground))
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
