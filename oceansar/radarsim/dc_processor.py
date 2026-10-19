#!/usr/bin/env python
"""Single-channel CDE/SDE processing of OceanSAR complex ``ProcFile`` SLCs.

Run ``python -m oceansar.radarsim.dc_processor --help`` for the CLI.
Detected InSAR/ATI L1b products cannot supply the required complex phase.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from oceansar.memory import memory_limited
from oceansar import constants
from oceansar import ocs_io
from oceansar.ocs_io import product_metadata as metadata
from oceansar.l2_processing_core import estimate_dc


def _scalar(reader, name, default=None):
    if name not in reader.__file__.variables:
        return default
    value = np.ma.asarray(reader.get(name))
    if value.size != 1 or np.ma.getmaskarray(value).any():
        return default
    result = float(value.item())
    return result if np.isfinite(result) else default


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be an integer')
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if integer != value:
        raise ValueError(f'{name} must be an integer')
    return integer


@memory_limited
def dc_process(cfg_file, proc_output_file, output_file=None, *, resolution=None,
               taper=None, channel=None, polarization=None, lag=None,
               pixel_spacing=None, min_pairs=None, min_correlation=None,
               valid_mask=None, segment_ids=None, nodata=None,
               window_samples=None, samples_per_resolution=None, plot=None,
               plot_show=False, plot_path=None, scansar_correction=None,
               crlb=None, crlb_snr_db=None):
    """Estimate single-channel Doppler centroid independently for selected SLCs.

    Parameters
    ----------
    cfg_file : str or path-like or None
        Optional configuration with a dc section. Non-None keyword arguments
        override its controls. With None, supply resolution explicitly.
    proc_output_file : str or path-like
        Complex ProcFile SLC with axes (channel, polarization, azimuth, range).
        Detected InSAR/ATI products cannot supply the required phase.
    output_file : str or path-like, optional
        Destination h5netcdf dataset. None returns results without saving a
        dataset. Must differ from the input SLC.
    resolution : float or pair of float, optional
        Requested second-moment averaging widths in metres (azimuth, ground
        range). Required explicitly or in dc.resolution. Integer support is
        chosen from the actual taper to meet these widths.
    taper : str, tuple, callable or 2D array, optional
        Nonnegative averaging weights on lag-pair midpoints, as accepted by
        estimate_dc. Default is 'boxcar'. Callable/array tapers require
        window_samples. Weights apply to correlations, not SLC amplitudes.
    channel : int or 'all', optional
        Zero-based channel index, default 0. 'all' estimates each channel
        independently and adds a channel dimension to the result.
    polarization : int or str, optional
        Zero-based index or configured label, such as 'vv'. None processes all
        available polarizations unless selected in the configuration.
    lag : int, optional
        Positive azimuth separation of correlated samples; default 1.
    pixel_spacing : float or pair of float, optional
        Input sampling in metres (azimuth, ground range). Otherwise inferred
        from SLC ground speed, azimuth sample rate, range sampling and incidence
        angle, using a local constant-incidence ground-range approximation.
    min_pairs : int, optional
        Minimum valid lag pairs for an estimate; default 1.
    min_correlation : float, optional
        Lower quality threshold for normalized CDE coherence or SDE correlation
        magnitude; default 0. Estimates at or below the threshold are NaN.
    valid_mask : 2D bool array, optional
        Valid input pixels, shape (azimuth, range), shared by selected SLCs.
        File fill masks and nonfinite samples are excluded independently.
    segment_ids : 1D array, optional
        Azimuth continuity labels. Correlation pairs never cross a label change.
    nodata : scalar, optional
        Explicit invalid sample value. Zero is otherwise valid.
    window_samples : pair of int, optional
        Explicit SLC support (azimuth, range), including lag extra azimuth
        rows. Its effective width must meet resolution.
    samples_per_resolution : float or pair of float, optional
        Requested output samples per effective width, default 2. Increase for
        denser maps with unchanged averaging resolution. At least 1; realized
        strides are integer input pixels, with a minimum of one pixel.
    plot : bool, optional
        Save diagnostic PNGs; defaults to dc.plot_save, or False. Requires
        output_file to determine the output directory.
    plot_show : bool, optional
        Display maps inline as well as optionally saving them; default False.
    plot_path : str or path-like, optional
        Figure directory, default dc.plot_path or 'dc_plots'. Relative paths
        are resolved beside output_file.
    scansar_correction : bool, optional
        Apply the ADPM finite-lag response correction in ScanSAR mode;
        defaults to dc.scansar_correction or True. Raw dc/dc_mean remain
        available. Corrected products are dc_shift_corrected and
        radial_velocity_corrected, with model baseline, sensitivity and
        inverse scaling factor. dc.correction_min_sensitivity (default .01),
        correction_min_correlation (.05) and correction_max_shift_hz (50)
        control validity. Stripmap estimates are unchanged.
    crlb : bool, optional
        Annotate ScanSAR products with the nominal unequalized acquisition
        CR lower bound; defaults to dc.crlb or True. Requires positive range
        bandwidth and incidence metadata. Missing range metadata is reported
        in dc_crlb_status. The local Whittle approximation uses the full
        input window, independent range cells and unknown homogeneous power;
        it does not claim that CDE/SDE attains the bound. Stripmap automatic
        annotation is not yet implemented.
    crlb_snr_db : float, optional
        Peak nominal antenna signal PSD / known white noise PSD, in dB.
        Defaults to dc.crlb_snr_db or None (speckle-only limit). It is not
        inferred from NESZ or measured/equalized SLC brightness.

    Returns
    -------
    xarray.Dataset
        CDE and SDE dc maps in Hz with axes (polarization, estimator, azimuth,
        range), plus channel when channel='all'. Coordinates are window centres
        in metres: azimuth includes SLC az0; range is an offset from the first
        input ground-range pixel. dc_mean pools each valid whole-image pair
        once; dc_std is the population spatial standard deviation of local
        maps. Valid carrier metadata additionally supplies radial_velocity,
        radial_velocity_mean and radial_velocity_std in m/s. Attributes report
        integer support, effective resolution and realized output spacing.
        ScanSAR correction additionally returns a baseline phase frequency,
        dimensionless sensitivity g and scale 1/g, corrected shift maps in
        Hz, corrected radial velocities in m/s, spatial statistics and
        correction-validity/reason flags. Raw pooled dc_mean stays uncorrected.
        ScanSAR CRB annotation adds dc_crlb_std in Hz and, with a carrier,
        radial_velocity_crlb_std in m/s. These have no estimator dimension;
        they bound physical Doppler directly, without another 1/g correction.
        Counts, model assumptions and validity flags accompany the prediction.

    Notes
    -----
    The observable is the stored SLC's phase frequency modulo fs_az / lag,
    where fs_az is az_sampling or, if absent, prf. No reramping, basebanding,
    phase-reference correction or ambiguity resolution changes the raw dc.
    ScanSAR response calibration is added separately when enabled.
    Equivalent radial velocity is -c * dc / (2 * f0). dc_mean is unweighted
    and is not the arithmetic mean of overlapping local estimates.
    """
    cfg = ocs_io.ConfigFile(str(cfg_file)) if cfg_file is not None else None
    controls = getattr(cfg, 'dc', None)
    def resolve(value, name, default=None):
        return value if value is not None else getattr(controls, name, default)
    resolution = resolve(resolution, 'resolution')
    if resolution is None:
        raise ValueError('Supply effective resolution in metres or set [dc] resolution')
    taper = resolve(taper, 'taper', 'boxcar')
    samples_per_resolution = resolve(samples_per_resolution, 'samples_per_resolution', 2.)
    if isinstance(taper, str) and taper.lower() == 'none':
        taper = None
    channel = resolve(channel, 'channel', 0)
    all_channels = isinstance(channel, str) and channel.lower() == 'all'
    if not all_channels:
        channel = _integer(int(channel) if isinstance(channel, str) and channel.isdigit() else channel,
                           'channel')
    lag = _integer(resolve(lag, 'lag', 1), 'lag')
    min_pairs = _integer(resolve(min_pairs, 'min_pairs', 1), 'min_pairs')
    min_correlation = resolve(min_correlation, 'min_correlation', 0.)
    polarization = resolve(polarization, 'polarization')
    pixel_spacing = resolve(pixel_spacing, 'pixel_spacing')
    nodata = resolve(nodata, 'nodata')
    window_samples = resolve(window_samples, 'window_samples')
    if window_samples is not None:
        window_samples = tuple(_integer(v, 'window_samples') for v in window_samples)
    proc_output_file = Path(proc_output_file)
    if output_file is not None and Path(output_file).resolve() == proc_output_file.resolve():
        raise ValueError('Output must not overwrite the input SLC')
    reader = ocs_io.ProcFile(str(proc_output_file), 'r')
    try:
        variables = reader.__file__.variables
        if 'slc_r' not in variables or 'slc_i' not in variables:
            raise ValueError('DC requires a complex ProcFile SLC, not a detected InSAR/ATI L1b file')
        real, imag = variables['slc_r'], variables['slc_i']
        if real.shape != imag.shape or len(real.shape) != 4:
            raise ValueError('Expected SLC shape (channel, polarization, azimuth, range)')
        nch, npol, naz, nrg = real.shape
        if not all_channels and not 0 <= channel < nch:
            raise ValueError(f'channel must be between 0 and {nch-1}')
        channels = list(range(nch)) if all_channels else [channel]
        labels = metadata.polarizations(reader.__file__, npol, cfg)
        if polarization is None:
            selected = list(range(npol))
        elif isinstance(polarization, str) and not polarization.isdigit():
            if polarization.lower() not in labels:
                raise ValueError(f'Unknown polarization {polarization!r}; available labels: {labels}')
            selected = [labels.index(polarization.lower())]
        else:
            index = _integer(int(polarization) if isinstance(polarization, str) else polarization,
                             'polarization')
            if not 0 <= index < npol:
                raise ValueError(f'polarization must be between 0 and {npol-1}')
            selected = [index]
        fs_az = metadata.azimuth_sampling(reader.__file__)
        if fs_az is None or fs_az <= 0:
            raise ValueError('SLC must provide a positive az_sampling or prf sample rate')
        v_ground = _scalar(reader, 'v_ground')
        rg_sampling = _scalar(reader, 'rg_sampling')
        incidence_deg = _scalar(reader, 'inc_angle')
        if pixel_spacing is None:
            if v_ground is None or v_ground <= 0 or rg_sampling is None or rg_sampling <= 0:
                raise ValueError('SLC requires positive v_ground and rg_sampling, or pixel_spacing override')
            if incidence_deg is None or not 0 < incidence_deg < 90:
                raise ValueError('SLC incidence angle must be in (0, 90) degrees, or override pixel_spacing')
            pixel_spacing = (v_ground/fs_az,
                             constants.c/(2*rg_sampling*np.sin(np.deg2rad(incidence_deg))))
            spacing_source = 'SLC v_ground, sampling and incidence metadata'
        else:
            spacing_source = 'explicit pixel_spacing override'
        az0 = _scalar(reader, 'az0', 0.)
        operation_mode = metadata.acquisition_mode(reader.__file__, cfg)
        correction = resolve(scansar_correction, 'scansar_correction', True)
        if not isinstance(correction, (bool, np.bool_)):
            raise ValueError('scansar_correction must be boolean')
        bound = resolve(crlb, 'crlb', True)
        if not isinstance(bound, (bool, np.bool_)):
            raise ValueError('crlb must be boolean')
        bound_snr = resolve(crlb_snr_db, 'crlb_snr_db', None)
        if isinstance(bound_snr, str) and bound_snr.strip().lower() == 'none':
            bound_snr = None
        bandwidth = _scalar(reader, 'rg_bw', getattr(getattr(cfg, 'mode', None), 'rg_bw', None))
        range_resolution = (constants.c/(2*bandwidth*np.sin(np.deg2rad(incidence_deg)))
            if bandwidth is not None and bandwidth > 0 and incidence_deg is not None
            and 0 < incidence_deg < 90 else None)
        bound_status = ('disabled' if not bound else 'not implemented for this acquisition mode'
            if operation_mode != 'scansar' else 'missing positive range bandwidth/incidence metadata'
            if range_resolution is None else 'annotated: approximate nominal-acquisition floor')
        model = None
        if operation_mode == 'scansar' and (correction or (bound and range_resolution is not None)):
            from oceansar.radarsim.scansar_dc import scansar_response_model
            model = scansar_response_model(cfg, reader)
        channel_results = []
        for channel_index in channels:
            results = []
            for index in selected:
                # Read only the selected channel/polarization, preserving fill masks.
                r = np.ma.asarray(real[channel_index, index, :, :])
                q = np.ma.asarray(imag[channel_index, index, :, :])
                valid = ~(np.ma.getmaskarray(r) | np.ma.getmaskarray(q))
                if valid_mask is not None:
                    supplied_mask = np.asarray(valid_mask, bool)
                    if supplied_mask.shape != (naz, nrg):
                        raise ValueError('valid_mask must match selected SLC image shape')
                    valid &= supplied_mask
                slc = np.asarray(r.filled(0), dtype=np.float64) + 1j*np.asarray(q.filled(0), dtype=np.float64)
                result = estimate_dc(slc, fs_az=fs_az, pixel_spacing=pixel_spacing,
                                     resolution=resolution, taper=taper, lag=lag,
                                     valid_mask=valid, segment_ids=segment_ids, nodata=nodata,
                                     min_pairs=min_pairs, min_correlation=min_correlation,
                                     origin=(az0, 0.), window_samples=window_samples,
                                     samples_per_resolution=samples_per_resolution)
                model_valid = valid & np.isfinite(slc)
                if nodata is not None:
                    model_valid &= slc != nodata
                if model is not None and correction:
                    from oceansar.radarsim.scansar_dc import calibrate_scansar_dc
                    result = calibrate_scansar_dc(result, model, model_valid, taper=taper,
                        segment_ids=segment_ids,
                        min_sensitivity=getattr(controls, 'correction_min_sensitivity', .01),
                        min_correlation=getattr(controls, 'correction_min_correlation', .05),
                        max_shift_hz=getattr(controls, 'correction_max_shift_hz', 50.))
                if model is not None and bound and range_resolution is not None:
                    from oceansar.radarsim.scansar_dc import annotate_scansar_crlb
                    result = annotate_scansar_crlb(result, model, model_valid,
                        range_resolution_m=range_resolution, segment_ids=segment_ids, snr_db=bound_snr)
                results.append(result.expand_dims(polarization=[labels[index]]))
            pol_result = xr.concat(results, dim='polarization', join='exact')
            channel_results.append(pol_result.expand_dims(channel=[channel_index]) if all_channels else pol_result)
        ds = xr.concat(channel_results, dim='channel', join='exact') if all_channels else channel_results[0]
        ds = ds.assign_coords(polarization_index=('polarization', selected))
        ds.attrs.update(source_slc=str(proc_output_file.resolve()), channel_index='all' if all_channels else channel,
                        input_channel_count=nch, spacing_source=spacing_source,
                        acquisition_mode=operation_mode,
                        dc_crlb_status=bound_status,
                        range_geometry='local ground range offset',
                        slc_phase_reference='as stored; no implicit phase correction',
                        azimuth_origin_source='SLC az0 metadata' if _scalar(reader, 'az0') is not None else 'local zero')
        for name in ('sr0', 'inc_angle', 'f0', 'v_ground', 'v_orbit', 'orbit_alt', 'prf', 'az_sampling', 'rg_sampling', 'reramp'):
            value = _scalar(reader, name)
            if value is not None:
                ds.attrs[f'input_{name}'] = value
        ds['dc_std'] = ds.dc.std(('azimuth', 'range'), skipna=True)
        ds.dc_std.attrs.update(units='Hz', long_name='Spatial standard deviation of local DC estimates')
        carrier = _scalar(reader, 'f0')
        if carrier is not None and carrier > 0:
            factor = -constants.c / (2 * carrier)
            ds['radial_velocity'] = ds.dc * factor
            ds['radial_velocity_mean'] = ds.dc_mean * factor
            ds['radial_velocity_std'] = ds.dc_std * abs(factor)
            for name in ('radial_velocity', 'radial_velocity_mean'):
                ds[name].attrs.update(units='m/s',
                    long_name='Equivalent radial velocity of stored-SLC phase',
                    conversion='v_r = -c * f_D / (2 * f0); no phase-reference correction')
            ds.radial_velocity_std.attrs.update(units='m/s',
                long_name='Spatial standard deviation of local equivalent radial velocities')
            ds.attrs['doppler_velocity_convention'] = 'v_r = -c * f_D / (2 * f0)'
            if 'dc_crlb_std' in ds:
                ds['radial_velocity_crlb_std'] = ds.dc_crlb_std*abs(factor)
                ds.radial_velocity_crlb_std.attrs.update(units='m/s',
                    long_name='Approximate nominal-acquisition radial velocity CR lower bound standard deviation',
                    interpretation='Optimal unbiased estimator floor, not estimated actual error',
                    ancillary_variables='dc_crlb_valid dc_crlb_reason')
            if 'dc_shift_corrected' in ds:
                for suffix in ('', '_mean', '_std'):
                    name = f'radial_velocity_corrected{suffix}'
                    ds[name] = ds[f'dc_shift_corrected{suffix}'] * (abs(factor) if suffix == '_std' else factor)
                    ds[name].attrs.update(units='m/s',
                        long_name='Radial velocity from ADPM-corrected acquisition Doppler shift',
                        conversion='v_r = -c * delta_f / (2 * f0); reference Doppler shift zero')
        if 'sin_az' in variables:
            direction = np.ma.asarray(reader.get('sin_az'))
            if direction.shape == (naz,) and not np.ma.getmaskarray(direction).any():
                row = (ds.azimuth.values-az0)/ds.attrs['pixel_spacing_m'][0]
                ds['look_azimuth_sine'] = ('azimuth', np.interp(row, np.arange(naz), direction))
    finally:
        reader.close()
    if output_file is not None:
        destination = Path(output_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(destination, engine='h5netcdf')
    plot = resolve(plot, 'plot_save', False)
    if plot or plot_show:
        if plot and output_file is None:
            raise ValueError('plot=True requires output_file for diagnostic PNG paths')
        destination = None
        if plot:
            destination = Path(resolve(plot_path, 'plot_path', 'dc_plots'))
            if not destination.is_absolute():
                destination = Path(output_file).parent / destination
        if 'dc_sensitivity' in ds:
            from oceansar.radarsim.doppler_plots import plot_dc_scaling
            plot_dc_scaling(ds, destination, show=plot_show)
        plot_dc(ds, destination, show=plot_show)
    return ds


def plot_dc(ds, plot_path=None, *, show=False, bounds=None):
    """Plot Doppler-centroid and equivalent radial-velocity maps.

    Parameters
    ----------
    ds : xarray.Dataset
    dc_process result with estimator/polarization dimensions, metre-valued
        azimuth/range coordinates and a channel_index attribute; an optional
        channel dimension is processed recursively. Velocity maps are drawn
        when radial_velocity is present.
    plot_path : str or path-like, optional
        Destination directory for PNGs, created when needed. None skips saving.
    show : bool, optional
        Display each figure, for example inline in a notebook; default False.
    bounds : dict, optional
        'range' and 'azimuth' display limits as (min, max) pairs in metres.
        Missing limits use the dataset extent.

    Returns
    -------
    None
        Save and/or display one figure per channel, polarization, estimator and
        available quantity. Figures use bwr with symmetric limits about zero
        and are closed after use.
    """
    import matplotlib.pyplot as plt
    from oceansar.radarsim.doppler_plots import symmetric_limit, map_plot_bounds, set_map_bounds, dc_plot_data
    ds = dc_plot_data(ds)
    destination = Path(plot_path) if plot_path is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    bounds = map_plot_bounds(ds, bounds)
    if 'channel' in ds.dims:
        for channel in ds.channel.values:
            selected = ds.sel(channel=channel, drop=True).copy(deep=False)
            selected.attrs = dict(ds.attrs, channel_index=int(channel))
            plot_dc(selected, destination, show=show, bounds=bounds)
        return
    channel = ds.attrs['channel_index']
    corrected = ds.attrs.get('dc_plot_corrected', False)
    quantities = [('dc', 'dc_mean', 'Corrected Doppler shift' if corrected else 'Stored-SLC phase frequency', 'Hz', '')]
    if 'radial_velocity' in ds:
        quantities.append(('radial_velocity', 'radial_velocity_mean',
                           'Equivalent radial velocity', 'm/s', '_rvel'))
    for pol in ds.polarization.values:
        for method in ds.estimator.values:
            for variable, mean_variable, label, unit, suffix in quantities:
                figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
                values = ds[variable].sel(polarization=pol, estimator=method)
                limit = symmetric_limit(values)
                mesh = axis.pcolormesh(ds.range, ds.azimuth, values,
                                       shading='auto', cmap='bwr', vmin=-limit, vmax=limit)
                figure.colorbar(mesh, ax=axis, label=f'{label} ({unit})')
                mean = float(ds[mean_variable].sel(polarization=pol, estimator=method).item())
                axis.set(xlabel='Ground range offset (m)', ylabel='Azimuth (m)',
                         title=f'Channel {channel} {pol} {method.upper()} — mean {mean:.2f} {unit}')
                set_map_bounds(axis, bounds)
                if destination is not None:
                    figure.savefig(destination / f'dc_ch{channel}_{pol}_{method}{suffix}.png', dpi=150)
                if show:
                    plt.show()
                plt.close(figure)



def main(argv=None):
    """Run DC estimation from the command line and print summary statistics.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments excluding the program name; None uses sys.argv. --cfg_file
        supplies defaults, --proc_file/--output_file select products, and
        --resolution, --taper, --channel, --polarization, --lag,
        --samples-per-resolution and --plot override dc controls. Product
        paths otherwise come from sim.path/proc_file/dc_file, with the output
        filename defaulting to dc_data.nc.

    Returns
    -------
    int
        Zero after writing the dataset and printing Doppler/velocity means,
        spatial standard deviations, resolution and spacing. Parsing failures
        and caught ValueError/OSError failures raise SystemExit through argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--cfg_file', type=Path)
    parser.add_argument('-p', '--proc_file', type=Path)
    parser.add_argument('-o', '--output_file', type=Path)
    parser.add_argument('--resolution', type=float, nargs=2, metavar=('AZ_M', 'GR_M'))
    parser.add_argument('--taper', help='Named SciPy taper, e.g. boxcar, hann or hamming')
    parser.add_argument('--channel', help='Zero-based channel index or all')
    parser.add_argument('--polarization', help='Zero-based index or configured label, e.g. vv')
    parser.add_argument('--lag', type=int)
    parser.add_argument('--samples-per-resolution', type=float, nargs=2, metavar=('AZ', 'GR'))
    parser.add_argument('--plot', action='store_true', default=None)
    parser.add_argument('--no-scansar-correction', dest='scansar_correction', action='store_false', default=None)
    parser.add_argument('--no-crlb', dest='crlb', action='store_false', default=None)
    parser.add_argument('--crlb-snr-db', type=float)
    args = parser.parse_args(argv)
    cfg = ocs_io.ConfigFile(str(args.cfg_file)) if args.cfg_file else None
    simulation = getattr(cfg, 'sim', None)
    root = Path(getattr(simulation, 'path', '.'))
    source = args.proc_file
    if source is None and simulation is not None:
        source = root / getattr(simulation, 'proc_file', 'proc_data.nc')
    if source is None:
        parser.error('Supply --proc_file or [sim] path/proc_file configuration')
    destination = args.output_file or root / getattr(simulation, 'dc_file', 'dc_data.nc')
    try:
        result = dc_process(args.cfg_file, source, destination, resolution=args.resolution,
                            taper=args.taper, channel=args.channel, polarization=args.polarization,
                            lag=args.lag, samples_per_resolution=args.samples_per_resolution, plot=args.plot,
                            scansar_correction=args.scansar_correction,
                            crlb=args.crlb, crlb_snr_db=args.crlb_snr_db)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(result[["dc_mean", "dc_std"]])
    if "radial_velocity_mean" in result:
        print(result[["radial_velocity_mean", "radial_velocity_std"]])
    if 'dc_shift_corrected' in result:
        print('ADPM-corrected Doppler shift spatial statistics (Hz):')
        print(result[['dc_shift_corrected_mean', 'dc_shift_corrected_std']])
    if 'dc_crlb_std' in result:
        floor = result.dc_crlb_std
        print('Nominal acquisition CRB std: min/median/max (Hz), optimal-estimator floor:')
        print(tuple(float(value.item()) for value in
                    (floor.min(skipna=True), floor.median(skipna=True), floor.max(skipna=True))))
        print('This is not the measured CDE/SDE error or the uncertainty of a spatial/channel average.')
    print(f'Effective resolution (m): {result.attrs["effective_resolution_m"]}')
    print(f'SLC support (m): {result.attrs["window_support_m"]}')
    print(f'Output spacing (m): {result.attrs["output_spacing_m"]}')
    print(f'Saved {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
