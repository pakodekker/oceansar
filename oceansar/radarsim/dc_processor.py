#!/usr/bin/env python
"""Single-channel CDE/SDE processing of OceanSAR complex ``ProcFile`` SLCs.

Run ``python -m oceansar.radarsim.dc_processor --help`` for the CLI.
Detected InSAR/ATI L1b products cannot supply the required complex phase.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from oceansar import constants
from oceansar import ocs_io
from oceansar.radarsim.dc_estimator import estimate_dc


def _scalar(reader, name, default=None):
    if name not in reader.__file__.variables:
        return default
    value = np.ma.asarray(reader.get(name))
    if value.size != 1 or np.ma.getmaskarray(value).any():
        return default
    result = float(value.item())
    return result if np.isfinite(result) else default


def _polarizations(cfg, count):
    pol = getattr(getattr(cfg, 'mode', None), 'pol', None)
    if pol is not None:
        labels = ['hh', 'vv'] if str(pol).lower() == 'dp' else [str(pol).lower()]
        if len(labels) != count:
            raise ValueError('Configured polarization count does not match the SLC')
        return labels
    return [f'pol{i}' for i in range(count)]


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


def dc_process(cfg_file, proc_output_file, output_file=None, *, resolution=None,
               taper=None, channel=None, polarization=None, lag=None,
               pixel_spacing=None, min_pairs=None, min_correlation=None,
               valid_mask=None, segment_ids=None, nodata=None,
               window_samples=None, samples_per_resolution=None, plot=None,
               plot_show=False, plot_path=None):
    """Estimate DC for one channel, independently for its selected polarizations.

    ``cfg_file`` may be None if the input path and resolution are explicit.
    Read resolution/taper/lag/channel/thresholds from an optional [dc] section;
    keyword values override configuration. Missing resolution raises an error.
    The default channel is 0; channel="all" processes channels independently and
    adds a channel dimension. polarization=None processes every polarization
    in each selected channel, or supply a zero-based index or a configured label.
    ``ProcFile`` metadata supplies the SLC azimuth sample rate (az_sampling,
    otherwise prf), ground speed, range sampling and incidence angle in degrees.
    Spatial spacing is (v_ground/fs_az, c/(2*rg_sampling*sin(incidence))).
    This is a local, constant-incidence ground-range approximation; calibrated
    pixel_spacing can override it. Azimuth coordinates include az0 when known.
    Range coordinates are local ground-range offsets from the first pixel.

    Return an xarray Dataset, with an additional polarization dimension.
    Write it to output_file using h5netcdf when specified. Optional diagnostic
    PNGs are written under [dc] plot_path relative to that output directory.
    plot_show=True also displays the maps inline; [dc] plot_save controls saving
    when plot is not explicitly supplied. No ocean state or second channel is
    required. No reramping, basebanding, phase-reference correction, ambiguity
    resolution is applied implicitly. When f0 metadata is valid, equivalent radial
    velocity is also returned as -c*dc/(2*f0), retaining the stored phase reference. dc_mean remains
    the unweighted pooled full-image estimate for each polarization.
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
        labels = _polarizations(cfg, npol)
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
        fs_az = _scalar(reader, 'az_sampling', _scalar(reader, 'prf'))
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
                results.append(result.expand_dims(polarization=[labels[index]]))
            pol_result = xr.concat(results, dim='polarization', join='exact')
            channel_results.append(pol_result.expand_dims(channel=[channel_index]) if all_channels else pol_result)
        ds = xr.concat(channel_results, dim='channel', join='exact') if all_channels else channel_results[0]
        ds = ds.assign_coords(polarization_index=('polarization', selected))
        ds.attrs.update(source_slc=str(proc_output_file.resolve()), channel_index='all' if all_channels else channel,
                        input_channel_count=nch, spacing_source=spacing_source,
                        range_geometry='local ground range offset',
                        slc_phase_reference='as stored; no implicit phase correction',
                        azimuth_origin_source='SLC az0 metadata' if _scalar(reader, 'az0') is not None else 'local zero')
        for name in ('sr0', 'inc_angle', 'f0', 'v_ground', 'v_orbit', 'orbit_alt', 'prf', 'rg_sampling', 'reramp'):
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
        plot_dc(ds, destination, show=plot_show)
    return ds


def plot_dc(ds, plot_path=None, *, show=False, bounds=None):
    """Plot CDE/SDE maps; optionally save PNGs and display in a notebook."""
    import matplotlib.pyplot as plt
    from oceansar.radarsim.doppler_plots import symmetric_limit, map_plot_bounds, set_map_bounds
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
    quantities = [('dc', 'dc_mean', 'Stored-SLC phase frequency', 'Hz', '')]
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
                            lag=args.lag, samples_per_resolution=args.samples_per_resolution, plot=args.plot)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(result[["dc_mean", "dc_std"]])
    if "radial_velocity_mean" in result:
        print(result[["radial_velocity_mean", "radial_velocity_std"]])
    print(f'Effective resolution (m): {result.attrs["effective_resolution_m"]}')
    print(f'SLC support (m): {result.attrs["window_support_m"]}')
    print(f'Output spacing (m): {result.attrs["output_spacing_m"]}')
    print(f'Saved {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
