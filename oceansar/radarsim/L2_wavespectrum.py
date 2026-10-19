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

import numpy as np
import scipy as sp
import xarray as xr
import matplotlib.pyplot as plt
import drama.utils as drtls
from matplotlib import colors
from matplotlib.patches import Circle

from oceansar.memory import memory_limited
from oceansar import ocs_io as tpio
from oceansar.ocs_io import product_metadata as metadata
from oceansar.utils.plotting import finish_plot as finish_figure
from oceansar import constants as const
from oceansar.radarsim.xspec_compression import save_compressed_xspec
from oceansar.l2_processing_core.cross_spectrum import (
    outlier_removal, welch_xspec as _welch_xspec,
)
from oceansar.surfaces import OceanSurface

__author__ = "Paco Lopez Dekker"
__email__ = "F.LopezDekker@tudeft.nl"


def generate_sublook_data(proc_data, sublook_filter, rg_ml, az_ml,
                          az_slice=slice(None), rg_slice=slice(None)):
    """Generate a multilooked intensity image for one Doppler sublook.

    Parameters
    ----------
    proc_data : ndarray
        Complex SLC data with dimensions ``(channel, azimuth, range)``.
    sublook_filter : ndarray
        One-dimensional Doppler-domain weighting function.
    rg_ml, az_ml : int
        Range and azimuth multilook window lengths.
    az_slice, rg_slice : slice
        Valid-region and downsampling selections for the output.

    Returns
    -------
    ndarray
        Multilooked sublook intensity data.
    """
    az_spectrum = sp.fft.fft(proc_data, axis=1)
    filter_shape = (1, proc_data.shape[1], 1)
    sublook_data = sp.fft.ifft(
        az_spectrum * sublook_filter.reshape(filter_shape), axis=1)
    sublook_data = np.abs(sublook_data)**2

    if rg_ml > 1 or az_ml > 1:
        for chind in range(sublook_data.shape[0]):
            if rg_ml > 1:
                sublook_data[chind] = drtls.smooth(
                    sublook_data[chind], rg_ml, axis=1)
            if az_ml > 1:
                sublook_data[chind] = drtls.smooth(
                    sublook_data[chind], az_ml, axis=0)

    return sublook_data[:, az_slice, rg_slice]


def wavespectrum_roi(image_shape, *, az0, pixel_spacing, scene_size,
                     average_azimuth_shift=0., azimuth_guard=0):
    """Locate the available scene within an SLC, then trim motion guards.

    Parameters
    ----------
    image_shape : pair of int
        Available SLC samples (azimuth, range).
    az0 : float
        Along-track coordinate of the first SLC row in metres.
    pixel_spacing, scene_size : pair of float
        Image spacing and requested scene size in metres (azimuth, range).
        Ground range starts at column zero; scene azimuth is centred on zero.
    average_azimuth_shift : float, optional
        Predicted mean motion displacement of the scene in metres.
    azimuth_guard : int, optional
        Samples discarded at both available azimuth edges for velocity spread.

    Returns
    -------
    azimuth_slice, range_slice : slice
        Bounded, nonempty selections with at least two pixels per axis.

    Raises
    ------
    ValueError
        Geometry is invalid or no usable region remains after clipping/guards.
    """
    shape = np.asarray(image_shape)
    spacing, scene = np.asarray(pixel_spacing, float), np.asarray(scene_size, float)
    if (shape.shape != (2,) or spacing.shape != (2,) or scene.shape != (2,)
            or not np.all(np.isfinite(shape)) or np.any(shape < 2)
            or np.any(shape != shape.astype(int))
            or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)
            or not np.all(np.isfinite(scene)) or np.any(scene <= 0)
            or not np.isfinite(az0) or not np.isfinite(average_azimuth_shift)
            or not np.isfinite(azimuth_guard) or azimuth_guard < 0
            or azimuth_guard != int(azimuth_guard)):
        raise ValueError('Invalid wave-spectrum ROI geometry or azimuth guard')
    start = int(np.floor((-az0-scene[0]/2+average_azimuth_shift)/spacing[0]))
    stop = int(np.ceil((-az0+scene[0]/2+average_azimuth_shift)/spacing[0]))
    # A ScanSAR SLC can be shorter than the displaced scene. Clip before
    # applying guards: a negative NumPy slice start would otherwise wrap to
    # the end of the SLC and silently produce an empty sublook image.
    start = int(np.clip(start, 0, shape[0]))+int(azimuth_guard)
    stop = int(np.clip(stop, 0, shape[0]))-int(azimuth_guard)
    range_stop = min(int(np.ceil(scene[1]/spacing[1])), int(shape[1]))
    if stop-start < 2 or range_stop < 2:
        raise ValueError('No usable wave-spectrum ROI after clipping to the SLC and trimming motion guards')
    return slice(start, stop), slice(0, range_stop)


def welch_xspec(int_fwd, int_bwd, imgpar, remove_outliers=True,
                window_size=(2e3, 2e3), lp_scale=1e3, thresh=100.0,
                tile_normalization=True, tile_outlier_threshold=3,
                win_func=None, verbose=False):
    """Adapt OceanSAR intensity images to the shared Welch estimator.

    Parameters
    ----------
    int_fwd, int_bwd : 2D real array
        Detected intensities ordered (azimuth, ground range). These inputs
        are already detected and must not be squared again.
    imgpar : dict
        Input geometry: az_spacing and grg_spacing, both in metres.
    remove_outliers : bool, optional
        Remove bright intensity outliers before estimating; default True.
    window_size : float or pair of float, optional
        Tile support in metres (azimuth, ground range), default (2000, 2000).
    lp_scale, thresh : float, optional
        Outlier smoothing support in metres and relative-excess threshold.
    tile_normalization : bool, optional
        Divide centered tiles by their own means (default True), otherwise
        by the corresponding full-image means.
    tile_outlier_threshold : float, optional
        Maximum excess tile brightness in population standard deviations.
    win_func : callable, optional
        1D taper taking a sample count; default numpy.hamming.
    verbose : bool, optional
        Print tile counts.

    Returns
    -------
    kx, ky, average, tiles : numpy.ndarray
        Range/azimuth axes in cycles/m, complex mean spectrum (ky, kx),
        and individual spectra (azimuth_tile, range_tile, ky, kx).
        See l2_processing_core.cross_spectrum.welch_xspec for normalization,
        tile rejection and handling of zero normalization means.
    """
    return _welch_xspec(
        int_fwd, int_bwd, pixel_spacing=(imgpar['az_spacing'], imgpar['grg_spacing']),
        remove_outliers=remove_outliers, window_size=window_size,
        lp_scale=lp_scale, thresh=thresh, tile_normalization=tile_normalization,
        tile_outlier_threshold=tile_outlier_threshold, win_func=win_func,
        verbose=verbose)


def add_polar_grid(ax, center=None, max_radius=None, radii=None, angles=None,
                   color='w', lw=0.6, alpha=0.8, label_radii=False,
                   fontsize=8):
    """Overlay wavelength circles and direction spokes on a spectrum axis."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if center is None:
        center = ((x0 + x1) / 2., (y0 + y1) / 2.)
    if max_radius is None:
        max_radius = min(abs(x1 - x0), abs(y1 - y0)) / 2.
    if radii is None:
        radii = np.linspace(max_radius / 4., max_radius, 4)
    if angles is None:
        angles = np.linspace(0., 360., 4, endpoint=False)

    for radius in radii:
        ax.add_patch(Circle(
            center, radius, fill=False, edgecolor=color, linewidth=lw,
            alpha=alpha, zorder=10))
    for angle in angles:
        angle = np.deg2rad(angle)
        endpoint = (
            center[0] + max_radius * np.cos(angle),
            center[1] + max_radius * np.sin(angle))
        ax.plot(
            [center[0], endpoint[0]], [center[1], endpoint[1]],
            color=color, linewidth=lw, alpha=alpha, zorder=11)

    if label_radii:
        angle = np.pi / 2.
        for radius in radii:
            ax.text(
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                f'{1 / radius:.0f}', color=color, fontsize=fontsize,
                va='center', ha='left', zorder=12)
            angle -= np.pi / 8.


def plot_xspec(kx, ky, xspec, ref_xspec=None, vmax=10, vmin=None, avg=3, imag=True, ivmax=None, polar_grid=True, noise_thresh=3,kymax=0.05,kxmax=0.075,
               figsize=(9,4), title_suffix='', ltitle= None, rtitle=None, db=True,pmax=None, pcmap='hsv', p_levels=None):
    shape_fwd = xspec.shape
    fig, axs = plt.subplots(1,2, figsize=figsize, constrained_layout=True)
    #vmin=-30
    #norm = colors.Normalize(vmin=vmin, vmax=vmax
    # /np.abs(xspec[:,:]).max()
    if kymax is None:
        kymax = ky.max()
    ky_inds = np.where((ky >= -kymax) & (ky <= kymax))[0]
    if kxmax is None:
        kxmax = kx.max()
    kx_inds = np.where((kx >= -kxmax) & (kx <= kxmax))[0]
    if ref_xspec is None:
        ref_xspec = xspec
    ref_power = np.abs(ref_xspec)
    noisefloor = 10*np.log10(np.nanmedian(ref_power[ky_inds,:][:,kx_inds]))
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = 10*np.log10(drtls.smooth(ref_power, 5)) - noisefloor
    if db:
        if vmin is None:
            vmin = noisefloor
        xspec_r = 10*np.log10(np.abs(xspec))
        cmap_r = 'viridis'
        color_pgrid = 'w'
    else:
        if vmin is None:
            vmin = np.min(xspec[ky_inds,:][:,kx_inds])
        xspec_r = xspec.real
        cmap_r = 'bwr'
        color_pgrid = 'k'
    im0 = axs[0].imshow(xspec_r[ky_inds][:,kx_inds], cmap=cmap_r, vmax=vmax, vmin=vmin, origin='lower', extent=[kx[kx_inds].min(), kx[kx_inds].max(), ky[ky_inds].min(), ky[ky_inds].max()])
    #fig.colorbar(im0, ax=axs[0], orientation='vertical')
    #print('hola')
    if imag:
        xspec_im = np.imag(xspec)
        if ivmax is None:
            ivmax = np.max(np.abs(xspec_im[ky_inds,:])) #2 * np.std(xspec_im[ky_inds,:])
            ivmax = np.min([ivmax,10])
        ivmin = -ivmax
        im1 = axs[1].imshow(
            xspec_im[ky_inds, :][:, kx_inds], cmap='bwr',
            origin='lower',
            extent=[kx[kx_inds].min(), kx[kx_inds].max(),
                    ky[ky_inds].min(), ky[ky_inds].max()],
            interpolation='nearest', vmin=ivmin, vmax=ivmax)
        if rtitle is None:
                rtitle = 'Imag Xspec' + title_suffix
        axs[1].set_title(rtitle)
        im2units=''
    else:
        xphase = np.angle(xspec)
        xphase[(snr < noise_thresh) | ~np.isfinite(snr)] = np.nan  # mask low SNR
        if pmax is None:
            pmax = np.pi
        im1_kwargs = {
            'cmap': pcmap,
            'origin': 'lower',
            'extent': [kx[kx_inds].min(), kx[kx_inds].max(),
                       ky[ky_inds].min(), ky[ky_inds].max()],
            'interpolation': 'nearest',
        }
        if p_levels is None:
            im1_kwargs.update(vmin=-pmax, vmax=pmax)
            cb2_ticks = None
        else:
            p_levels = np.asarray(p_levels, dtype=float)
            im1_kwargs['norm'] = colors.BoundaryNorm(p_levels, plt.get_cmap(pcmap, len(p_levels) - 1).N)
            im1_kwargs['cmap'] = plt.get_cmap(pcmap, len(p_levels) - 1)
            cb2_ticks = p_levels if len(p_levels) <= 12 else None
        im1 = axs[1].imshow(xphase[ky_inds,:][:,kx_inds], **im1_kwargs)
        if rtitle is None:
            rtitle = 'Xspec Phase' + title_suffix
        axs[1].set_title(rtitle)
        im2units = 'rad'
    if polar_grid:
        if kxmax < 0.02:
            radii_wl = [400, 200, 100]
        else:
            radii_wl = [200, 100,50, 25]
        radii= 1/ np.array(radii_wl)
        add_polar_grid(axs[0], color=color_pgrid, lw=0.5, alpha=0.7, radii=radii,label_radii=True)
        add_polar_grid(axs[1], color='k', lw=0.5, alpha=0.7, radii=radii,label_radii=True)
    #fig.colorbar(im1, ax=axs[1], orientation='vertical')
    # divider = make_axes_locatable(axs[0])
    # cax0 = divider.append_axes("right", size="3%", pad=0.05)
    #fig.colorbar(im0, cax=cax0, orientation='vertical')
    for ax in axs:
        ax.set_ylim(ky[ky_inds].min(), ky[ky_inds].max())
        ax.set_xlim(-kxmax, kxmax)
        #ax.set_xticks(np.arange(-0.08,0.12,0.04))
    # divider = make_axes_locatable(axs[1])
    # cax1 = divider.append_axes("right", size="3%", pad=0.05)
    # fig.colorbar(im1, cax=cax1, orientation='vertical')
    #cb1 = fig.colorbar(im0, ax=axs[0], orientation='vertical',  fraction=0.06, pad=0.04)
    cb1 = fig.colorbar(im0, ax=axs[0], orientation='vertical', shrink=0.4, fraction=0.03, pad=0.04)
    axs[1].tick_params(axis='y', labelleft=False)   # removes left-side labels on ax1
    axs[1].set_ylabel('')
    if ltitle is None:
        ltitle = 'Xspec Rel. Magnitude'
    axs[0].set_title(ltitle)
    if db:
        cb1.ax.set_xlabel('dB', labelpad=10)
    cb2 = fig.colorbar(im1, ax=axs[1],orientation='vertical',  shrink=0.4, fraction=0.03, pad=0.04, ticks=cb2_ticks if not imag else None)
    cb2.ax.set_xlabel(im2units, labelpad=10)
    return fig, axs

@memory_limited
def l2_wavespectrum(cfg_file, proc_output_file, ocean_file, output_file, xvmax=10, xvmin=None, xivmax=None, kymax=0.05, kxmax=0.075, *, plot_show=False):

    """Estimate wave cross-spectra from multilooked Doppler sublook intensities.

    Parameters
    ----------
    cfg_file : str or path-like
        Configuration supplying L2_wavespectrum sublook,
        multilook, Welch averaging, outlier-removal and plotting controls.
    proc_output_file : str or path-like
        Complex ProcFile SLC with axes (channel, polarization, azimuth, range).
    ocean_file : str or path-like
        Current-format ocean initial conditions used for reference spectra.
    output_file : str or path-like
        Destination compressed NetCDF dataset; a non-.nc extension is replaced
        by .nc. wxspec is complex with axes (sublook_pair, channel,
        polarization, ky, kx); sublook_mean stores mean intensities with axes
        (sublook, channel, polarization). kx/ky coordinates are cycles/m and
        channel baselines are in metres.
    xvmax : float, optional
        Upper display limit in dB for cross-spectrum magnitude; default 10.
    xvmin : float, optional
        Lower magnitude display limit in dB. None uses the plotting helper's
        noise-floor estimate.
    xivmax : float, optional
        Symmetric display limit for the imaginary cross-spectrum in linear
        cross-spectrum units. None selects limits automatically.
    kymax : float, optional
        Reserved y-wavenumber display limit in cycles/m. Currently not
        forwarded to the plotting helper.
    kxmax : float, optional
        Reserved x-wavenumber display limit in cycles/m. Currently not
        forwarded to the plotting helper.
    plot_show : bool, optional
        Display enabled figures while also saving them if configured. Enabled
        unsaved figures are displayed automatically. All figures are closed.

    Returns
    -------
    None
        Write the cross-spectrum dataset and configured figures.

    Raises
    ------
    ValueError
        The displaced scene has no usable overlap with the SLC after motion
        guards, or spectral inputs/controls are invalid.
    """
    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR L2 Wavespectra: [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-------------------------------------------------------------------')

    print('Initializing...')

    ## CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

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

    # PROCESSED DATA
    proc_content = tpio.ProcFile(proc_output_file, 'r')
    proc_data = proc_content.get('slc*')
    polt = metadata.polarizations(proc_content.__file__, proc_data.shape[1], cfg)
    acquisition_prf = metadata.scalar(proc_content.__file__, 'prf', positive=True)
    operation_mode = metadata.acquisition_mode(proc_content.__file__, cfg)
    proc_bw = getattr(proc_content.__file__, 'processed_doppler_bandwidth_hz', None)
    if proc_bw is None and 'dc_processing_frequency' in proc_content.__file__.variables:
        frequencies = proc_content.get('dc_processing_frequency')
        support = proc_content.get('dc_processing_power') > 0
        if np.count_nonzero(support) > 1:
            proc_bw = np.ptp(frequencies[support]) + np.median(np.diff(frequencies))
    if proc_bw is None:
        raise ValueError('SLC must record its processed Doppler bandwidth; refocus the raw data')
    if not np.isfinite(proc_bw) or proc_bw <= 0:
        raise ValueError('SLC processed Doppler bandwidth must be positive and finite')
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
    inc_angle = np.deg2rad(proc_content.get('inc_angle'))
    # print(f"Read inc_angle from proc file: {np.rad2deg(inc_angle)}")
    proc_content.close()
    # OCEAN SURFACE
    surface = OceanSurface()
    surface.load(ocean_file, compute=['V'])
    surface.t = 0.

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

    # SURFACE RADIAL VELOCITY
    v_radial_surf = surface.Vx*np.sin(inc_angle) - surface.Vz*np.cos(inc_angle)
    v_radial_surf_mean = np.mean(v_radial_surf)
    v_radial_surf_std = np.std(v_radial_surf)

    # Expected mean azimuth shift
    avg_az_shift = - v_radial_surf_mean / v_ground * sr0
    std_az_shift = v_radial_surf_std / v_ground * sr0


    print('Starting Wavespectrum processing...')

    # Only scene dimensions and velocity statistics are needed below. Release
    # the large Fourier/surface grids before generating all sublook images.
    rg_span, az_span = float(surface.Lx), float(surface.Ly)
    del surface, v_radial_surf
    rg_size = proc_data[0].shape[2]
    az_size = proc_data[0].shape[1]

    az_guard = int(std_az_shift / (v_ground / prf))
    az_roi, rg_roi = wavespectrum_roi(
        (az_size, rg_size), az0=float(az0),
        pixel_spacing=(float(az_grid_spacing), float(grg_grid_spacing)),
        scene_size=(az_span, rg_span), average_azimuth_shift=float(avg_az_shift),
        azimuth_guard=az_guard)
    az_min, az_max = az_roi.start, az_roi.stop
    rg_min, rg_max = rg_roi.start, rg_roi.stop
    print(f'Wave-spectrum SLC ROI: azimuth [{az_min}:{az_max}], range [{rg_min}:{rg_max}]')

    # Adaptive coregistration
    # all of this is not needed
    # if cfg.sar.L_total:
    #     ant_L = ant_L/float(num_ch)
    #     dist_chan = ant_L/2
    # else:
    #     if float(cfg.sar.Spacing) != 0:
    #         dist_chan = float(cfg.sar.Spacing)/2
    #     else:
    #         dist_chan = ant_L/2
    # # dist_chan = ant_L/num_ch/2.
    # print('ATI Spacing: %f' % dist_chan)
    # inter_chan_shift_dist = dist_chan/(v_ground/prf)
    # # Subsample shift in azimuth
    # for chind in range(proc_data.shape[0]):
    #     shift_dist = - chind * inter_chan_shift_dist
    #     shift_arr = np.exp(-2j * np.pi * shift_dist *
    #                        np.roll(np.arange(az_size) - az_size/2,
    #                                int(-az_size / 2)) / az_size)
    #     shift_arr = shift_arr.reshape((1, az_size, 1))
    #     proc_data[chind] = np.fft.ifft(np.fft.fft(proc_data[chind], axis=1) *
    #                                    shift_arr, axis=1)

    # First dimension is number of channels, second is number of pols
    ch_dim = proc_data.shape[0:2]
    npol = ch_dim[1]
    proc_data_rshp = [np.prod(ch_dim), proc_data.shape[2], proc_data.shape[3]]
    # Compute extended covariance matrices...
    proc_data = proc_data.reshape(proc_data_rshp)
    # Intensities
    i_all = []
    for chind in range(proc_data.shape[0]):
        this_i = drtls.smooth(
            drtls.smooth(
                np.abs(proc_data[chind])**2., rg_ml,
                axis=1, window=ml_win),
            az_ml, axis=0, window=ml_win)
        i_all.append(this_i[az_min:az_max, rg_min:rg_max])
    i_all = np.array(i_all)

    ## Wave spectra computation
    ## Processed Doppler bandwidth
    fa = sp.fft.fftfreq(proc_data_rshp[1], 1/prf)
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
    az_downsmp = max(1, int(np.floor(az_ml / 2)))
    rg_downsmp = max(1, int(np.floor(rg_ml / 2)))
    sublooks = []
    sublooks_mean = []
    az_slice = slice(az_min, az_max, az_downsmp)
    rg_slice = slice(rg_min, rg_max, rg_downsmp)
    delta_rg = rg_downsmp * grg_grid_spacing
    delta_az = az_downsmp * az_grid_spacing

    for i_sbl in range(n_sublook):
        sublook_data = generate_sublook_data(
            proc_data, sublook_filt[i_sbl], rg_ml, az_ml,
            az_slice=az_slice, rg_slice=rg_slice)
        sublooks.append(sublook_data)
        sublooks_mean.append(np.mean(sublook_data, axis=(1, 2)))

    imgpar = {
        'az_spacing': delta_az,
        'grg_spacing': delta_rg,
    }
    xspecs = []
    tind = 0
    xspec_lut = np.zeros((len(sublooks), len(sublooks)), dtype=int)
    sublook_pair_1 = []
    sublook_pair_2 = []
    kgrg = None
    kaz = None

    for ind1 in range(len(sublooks)):
        for ind2 in range(ind1 + 1, len(sublooks)):
            xspec_lut[ind1, ind2] = tind
            sublook_pair_1.append(ind1)
            sublook_pair_2.append(ind2)
            tind = tind + 1
            pair_xspecs = []
            for chind in range(sublooks[ind1].shape[0]):
                kgrg_, kaz_, xspec, tiles = welch_xspec(
                    sublooks[ind1][chind], sublooks[ind2][chind],
                    imgpar)
                if kgrg is None:
                    kgrg = kgrg_
                    kaz = kaz_
                pair_xspecs.append(xspec)
                del tiles
            xspecs.append(np.asarray(pair_xspecs))

    pair_count = len(xspecs)
    xspec_values = np.asarray(xspecs, dtype=np.complex64).reshape(
        pair_count, int(ch_dim[0]), int(ch_dim[1]),
        kaz.size, kgrg.size)
    sublook_mean_values = np.asarray(sublooks_mean).reshape(
        len(sublooks), int(ch_dim[0]), int(ch_dim[1]))
    xspec_dataset = xr.Dataset(
        data_vars={
            'wxspec': (
                ('sublook_pair', 'channel', 'polarization', 'ky', 'kx'),
                xspec_values),
            'sublook_mean': (
                ('sublook', 'channel', 'polarization'),
                sublook_mean_values),
            'b_ati': (('channel',), np.asarray(b_ati)),
            'b_xti': (('channel',), np.asarray(b_xti)),
        },
        coords={
            'sublook_pair': np.arange(pair_count),
            'sublook_1': ('sublook_pair', sublook_pair_1),
            'sublook_2': ('sublook_pair', sublook_pair_2),
            'sublook': np.arange(len(sublooks)),
            'channel': np.arange(int(ch_dim[0])),
            'polarization': np.asarray(polt),
            'kx': kgrg,
            'ky': kaz,
        },
        attrs={
            'title': 'OceanSAR sublook cross-spectra',
            'source_slc': os.path.abspath(proc_output_file),
            'incidence_angle': float(inc_angle),
            'radar_frequency': float(f0),
            'prf': float(acquisition_prf),
            'az_sampling': float(prf),
            'processed_doppler_bandwidth_hz': float(proc_bw),
            'acquisition_mode': operation_mode,
            'ground_velocity': float(v_ground),
            'range_pixel_spacing': float(delta_rg),
            'azimuth_pixel_spacing': float(delta_az),
            'sublook_count': int(n_sublook),
            'slc_roi_samples': [az_min, az_max, rg_min, rg_max],
        })
    xspec_dataset['wxspec'].attrs['long_name'] = (
        'Welch cross-spectrum of sublook intensities')
    xspec_dataset['kx'].attrs.update({
        'long_name': 'ground-range spatial frequency',
        'units': 'cycles m-1',
    })
    xspec_dataset['ky'].attrs.update({
        'long_name': 'azimuth spatial frequency',
        'units': 'cycles m-1',
    })
    xspec_dataset['b_ati'].attrs['units'] = 'm'
    xspec_dataset['b_xti'].attrs['units'] = 'm'

    output_root, output_ext = os.path.splitext(output_file)
    if output_ext.lower() != '.nc':
        output_file = output_root + '.nc'
    save_compressed_xspec(xspec_dataset, output_file)
    print(f"Saved compressed cross-spectra to {output_file}")


    image_extent = [(rg_min-.5)*grg_grid_spacing, (rg_max-.5)*grg_grid_spacing,
                    az0+(az_min-.5)*az_grid_spacing, az0+(az_max-.5)*az_grid_spacing]
    # PROCESSED AMPLITUDE
    if plot_proc_ampl:
        for pind in range(npol):
            save_path = (plot_path + os.sep + 'amp_dB_' + polt[pind]+
                         '.' + plot_format)
            plt.figure()
            plt.imshow(drtls.db(i_all[pind]), aspect='equal',
                       origin='lower',
                       vmin=drtls.db(np.max(i_all[pind]))-20,
                       extent=image_extent, interpolation='nearest',
                       cmap='viridis')
            plt.xlabel('Ground range [m]')
            plt.ylabel('Azimuth [m]')
            plt.title("Amplitude")
            plt.colorbar()
            finish_figure(plt.gcf(), save_path, save=plot_save, show=plot_show)

            save_path = (plot_path + os.sep + 'amp_' + polt[pind]+
                         '.' + plot_format)
            int_img = (i_all[pind])**0.5
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

    # Plot the first channel/polarization for each sublook pair.
    if plot_spectrum:
        for ind1 in range(len(sublooks)):
            for ind2 in range(ind1 + 1, len(sublooks)):
                xspec = xspecs[xspec_lut[ind1, ind2]][0]
                if krg_ml > 1:
                    xspec = drtls.smooth(xspec, krg_ml, axis=1)
                if kaz_ml > 1:
                    xspec = drtls.smooth(xspec, kaz_ml, axis=0)
                fig, _ = plot_xspec(
                    kgrg, kaz, xspec, imag=True, vmax=xvmax, vmin=xvmin, ivmax=xivmax, avg=3, polar_grid=True,
                    title_suffix=f' ({ind1 + 1}, {ind2 + 1})')
                save_path = os.path.join(
                    plot_path,
                    f'xspec_{ind1 + 1}{ind2 + 1}.{plot_format}')
                finish_figure(fig, save_path, save=plot_save, show=plot_show)




if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-p', '--proc_file')
    parser.add_argument('-s', '--ocean_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    l2_wavespectrum(args.cfg_file, args.proc_file, args.ocean_file, args.output_file)
