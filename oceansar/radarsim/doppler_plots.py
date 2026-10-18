"""ATI maps and DC/ATI comparisons on their shared window-centre domain."""
from pathlib import Path

import numpy as np


def dc_plot_data(ds):
    """Return a plotting view using response-corrected ScanSAR maps if present.

    The input dataset and its raw pooled dc_mean are never modified. Corrected
    display means/std are spatial statistics of the corrected local maps.
    """
    if 'dc_shift_corrected' not in ds:
        return ds
    view = ds.copy(deep=False)
    view.attrs = dict(ds.attrs, dc_plot_corrected=True)
    for target, source in (('dc', 'dc_shift_corrected'),
                           ('radial_velocity', 'radial_velocity_corrected')):
        if source in ds:
            for suffix in ('', '_mean', '_std'):
                view[target+suffix] = ds[source+suffix]
    return view


def plot_dc_scaling(ds, plot_path=None, *, show=False):
    """Plot ScanSAR sensitivity, scaling and any nominal CRB across the burst.

    Parameters
    ----------
    ds : xarray.Dataset
        Corrected dc_process result with dc_sensitivity and dc_scale_factor.
        Select the first channel/polarization to avoid duplicate geometry plots.
    plot_path : str or path-like, optional
        Destination directory for dc_scansar_scaling.png; None skips saving.
    show : bool, optional
        Display the figure inline; default False. Close it after use.

    Returns
    -------
    None
        One figure with separate sensitivity/scaling panels, plus a nominal
        CRB standard-deviation panel when annotated. Curves show
        near, middle and far range window centres for both CDE and SDE.
        Invalid inverse factors remain gaps rather than being clipped.
    """
    if 'dc_sensitivity' not in ds:
        return
    import matplotlib.pyplot as plt
    selected = ds
    for dimension in ('channel', 'polarization'):
        if dimension in selected.dims:
            selected = selected.isel({dimension: 0}, drop=True)
    ranges = sorted(set((0, selected.sizes['range']//2, selected.sizes['range']-1)))
    has_bound = 'dc_crlb_std' in selected
    figure, axes = plt.subplots(3 if has_bound else 2, 1,
        figsize=(10, 10 if has_bound else 7), sharex=True, constrained_layout=True)
    colors = plt.get_cmap('tab10')
    for index, column in enumerate(ranges):
        for method_index, method in enumerate(selected.estimator.values):
            label = f'{str(method).upper()}, range {float(selected.range[column]):.0f} m'
            for axis, variable in zip(axes[:2], ('dc_sensitivity', 'dc_scale_factor')):
                axis.plot(selected.azimuth, selected[variable].sel(estimator=method).isel(range=column),
                          color=colors(index), linestyle='-' if method_index == 0 else '--', label=label)
        if has_bound:
            axes[2].plot(selected.azimuth, selected.dc_crlb_std.isel(range=column),
                color=colors(index), label=f'Range {float(selected.range[column]):.0f} m')
    axes[0].set_ylabel('Sensitivity g = d f̂ / d δ')
    axes[1].set_ylabel('Doppler correction factor 1/g')
    if has_bound:
        axes[2].set_ylabel('Nominal CRB std (Hz)')
        axes[2].set_title('Unequalized acquisition; approximate optimal unbiased estimator floor')
    axes[-1].set_xlabel('Azimuth position within burst image (m)')
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=9, ncol=2)
    axes[0].set_title('ScanSAR Doppler-centroid response: fixed burst, processing and antenna equalization')
    if plot_path is not None:
        destination = Path(plot_path)
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination/'dc_scansar_scaling.png', dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def symmetric_limit(values):
    """Finite maximum absolute value, with a usable range for zero/empty maps."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return (float(np.max(np.abs(finite))) if finite.size else 0.) or 1.


def coherence_lower_limit(values):
    """Pad the finite minimum by 0.25, clamp at zero, and round down to 0.1."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.
    lower = max(0., min(1., float(finite.min())) - .25)
    return float(np.floor(np.nextafter(lower*10, np.inf))/10)


def common_map_bounds(*datasets):
    """Intersection of output window-centre coverage, in metres."""
    bounds = {}
    for index, axis in enumerate(('azimuth', 'range')):
        limits = []
        for ds in datasets:
            values = ds[axis].values
            if not values.size or not np.all(np.isfinite(values)):
                raise ValueError(f'{axis} must contain finite output coordinates')
            lo, hi = float(values.min()), float(values.max())
            if lo == hi:
                spacing = ds.attrs.get('output_spacing_m', ds.attrs.get('pixel_spacing_m', (1., 1.)))[index]
                lo, hi = lo-spacing/2, hi+spacing/2
            limits.append((lo, hi))
        lo, hi = max(limit[0] for limit in limits), min(limit[1] for limit in limits)
        if hi <= lo:
            raise ValueError('DC and ATI window-centre domains do not overlap')
        bounds[axis] = (lo, hi)
    return bounds


def map_plot_bounds(ds, bounds=None):
    """Use the requested reference frame, retaining native samples and empty margins."""
    coverage = common_map_bounds(ds)
    if bounds is None:
        return coverage
    result = {}
    for axis in ('azimuth', 'range'):
        lo, hi = map(float, bounds[axis])
        if not np.isfinite(lo+hi) or hi <= lo:
            raise ValueError(f'Invalid {axis} display bounds')
        if hi < coverage[axis][0] or lo > coverage[axis][1]:
            raise ValueError(f'Display bounds do not overlap {axis} coordinates')
        result[axis] = (lo, hi)
    return result


def set_map_bounds(axis, bounds):
    """Identical limits, endpoint ticks and metre scaling for all estimated maps."""
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    axis.set_xlim(*bounds['range'])
    axis.set_ylim(*bounds['azimuth'])
    axis.set_aspect('equal', adjustable='box')
    for ticker, name in ((axis.xaxis, 'range'), (axis.yaxis, 'azimuth')):
        lo, hi = bounds[name]
        ticks = np.linspace(lo, hi, 3)
        ticks[np.abs(ticks) < 64*np.finfo(float).eps*max(abs(lo), abs(hi), 1.)] = 0.
        ticker.set_major_locator(FixedLocator(ticks))
        ticker.set_major_formatter(FormatStrFormatter('%.0f' if hi-lo >= 100 else '%g'))


def plot_ati_matrix(ds, plot_path=None, *, show=False, plot_format='png', bounds=None):
    """One matrix per polarization: phase above, coherence below, intensity on diagonal."""
    import matplotlib.pyplot as plt
    destination = Path(plot_path) if plot_path is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    channels = list(ds.channel.values)
    nch = len(channels)
    bounds = map_plot_bounds(ds, bounds)
    image_aspect = ((bounds['azimuth'][1]-bounds['azimuth'][0]) /
                    (bounds['range'][1]-bounds['range'][0]))
    for pol in ds.polarization.values:
        phases = ds.pair_phase.sel(polarization=pol)
        coherences = ds.pair_coherence.sel(polarization=pol)
        phase_limit = symmetric_limit(phases)
        coherence_min = coherence_lower_limit(coherences)
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity = 10*np.log10(ds.intensity.sel(polarization=pol).where(ds.intensity.sel(polarization=pol) > 0))
        finite_intensity = intensity.values[np.isfinite(intensity.values)]
        intensity_max = float(finite_intensity.max()) if finite_intensity.size else 0.
        intensity_min = intensity_max - 30.
        width = max(10., 2.5*nch+1.8)
        height = max(6., 2.5*nch*image_aspect + 2.2)
        figure, axes = plt.subplots(nch, nch, figsize=(width, height),
                                    sharex=True, sharey=True, squeeze=False,
                                    constrained_layout=True)
        phase_image = coherence_image = None
        for row, first in enumerate(channels):
            for column, last in enumerate(channels):
                axis = axes[row, column]
                axis.set_label(f'antenna_{first}_{last}')
                axis.set_aspect('equal', adjustable='box')
                if row < column:
                    values = phases.sel(pair=f'{first}-{last}')
                    phase_image = axis.pcolormesh(ds.range, ds.azimuth, values, shading='auto',
                                                 cmap='bwr', vmin=-phase_limit, vmax=phase_limit)
                    annotation = 'Phase'
                elif row > column:
                    values = coherences.sel(pair=f'{last}-{first}')
                    coherence_image = axis.pcolormesh(ds.range, ds.azimuth, values, shading='auto',
                                                     cmap='bone', vmin=coherence_min, vmax=1.)
                    annotation = 'Coherence'
                else:
                    axis.pcolormesh(ds.range, ds.azimuth, intensity.sel(channel=first), shading='auto',
                                    cmap='viridis', vmin=intensity_min, vmax=intensity_max)
                    annotation = 'Intensity'
                axis.text(.04, .95, annotation, transform=axis.transAxes, va='top', fontsize=16,
                          bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 2})
                axis.tick_params(labelsize=14)
                set_map_bounds(axis, bounds)
                if row == 0:
                    axis.set_title(f'Antenna {last}', fontsize=20)
                if column == 0:
                    axis.set_ylabel(f'Ant. {first}', fontsize=18, rotation=0, labelpad=45, va='center')
        figure.supylabel('Azimuth (m)', fontsize=18)
        figure.supxlabel('Ground range offset (m)', fontsize=18)
        if phase_image is not None:
            colorbar = figure.colorbar(phase_image, ax=axes.ravel().tolist(), orientation='vertical',
                                      fraction=.025, pad=.025)
            colorbar.set_label('Phase (rad)', fontsize=20)
            colorbar.ax.tick_params(labelsize=20)
        if coherence_image is not None:
            colorbar = figure.colorbar(coherence_image, ax=axes.ravel().tolist(), orientation='horizontal',
                                      fraction=.025, pad=.04)
            colorbar.set_label('Coherence', fontsize=20)
            colorbar.ax.tick_params(labelsize=20)
        figure.suptitle(f'ATI {pol}: phase/coherence/intensity matrix\n'
                       f'Diagonal intensity: {intensity_min:.1f} to {intensity_max:.1f} dB', fontsize=24)
        if destination is not None:
            figure.savefig(destination / f'ATI_pair_matrix_{pol}.{plot_format}', dpi=150)
        if show:
            plt.show()
        plt.close(figure)


def plot_dc_summary(ds, plot_path=None, *, show=False, bounds=None):
    """Display channel 0 and the arithmetic channel-average maps; return the average."""
    import matplotlib.pyplot as plt
    plot_dc_scaling(ds, plot_path, show=show)
    ds = dc_plot_data(ds)
    destination = Path(plot_path) if plot_path is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    bounds = map_plot_bounds(ds, bounds)
    quantities = [('dc', 'Hz')]
    if 'radial_velocity' in ds:
        quantities.append(('radial_velocity', 'm/s'))
    if 'channel' in ds.dims:
        channel = 0 if 0 in ds.channel.values else int(ds.channel.values[0])
        groups = [(f'Channel {channel}', ds.sel(channel=channel, drop=True))]
        average = ds[[name for name, unit in quantities]].mean('channel', skipna=True, keep_attrs=True)
        average.attrs.update(channel_index='average',
            averaging='Arithmetic mean of finite channel estimates per pixel; no phase ambiguity correction')
        for name, unit in quantities:
            average[f'{name}_mean'] = average[name].mean(('azimuth', 'range'), skipna=True)
            average[f'{name}_std'] = average[name].std(('azimuth', 'range'), skipna=True)
            for suffix in ('mean', 'std'):
                average[f'{name}_{suffix}'].attrs['units'] = unit
        if ds.sizes['channel'] > 1:
            groups.append((f'Average of {ds.sizes["channel"]} channels', average))
    else:
        average = ds
        groups = [(f'Channel {ds.attrs.get("channel_index", 0)}', ds)]
    for pol in ds.polarization.values:
        methods = list(ds.estimator.values)
        figure, axes = plt.subplots(len(quantities), len(groups)*len(methods),
                                    figsize=(4*len(groups)*len(methods), 3.5*len(quantities)),
                                    constrained_layout=True, squeeze=False)
        for row, (quantity, unit) in enumerate(quantities):
            fields, labels = [], []
            for label, group in groups:
                for method in methods:
                    fields.append(group[quantity].sel(polarization=pol, estimator=method))
                    labels.append(f'{label}\n{method.upper()} ({unit})')
            limit = symmetric_limit(np.concatenate([field.values.ravel() for field in fields]))
            for axis, field, label in zip(axes[row], fields, labels):
                mesh = axis.pcolormesh(ds.range, ds.azimuth, field, shading='auto',
                                       cmap='bwr', vmin=-limit, vmax=limit)
                axis.set(title=label, xlabel='Ground range offset (m)', ylabel='Azimuth (m)')
                set_map_bounds(axis, bounds)
            figure.colorbar(mesh, ax=axes[row].tolist(), label=unit)
        label = 'Corrected DC' if ds.attrs.get('dc_plot_corrected', False) else 'DC'
        figure.suptitle(f'{label} {pol}: channel 0 and channel average' if len(groups) > 1 else f'{label} {pol}')
        if destination is not None:
            figure.savefig(destination / f'dc_channel_summary_{pol}.png', dpi=150)
        if show:
            plt.show()
        plt.close(figure)
    return average


def plot_lag_summary(ds, plot_path=None, *, show=False, plot_format='png'):
    """Plot spatial means for every pair and the mean map at each antenna lag."""
    import matplotlib.pyplot as plt
    destination = Path(plot_path) if plot_path is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    for pol in ds.polarization.values:
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
        for axis, (quantity, unit) in zip(axes, [('dc', 'Hz'), ('radial_velocity', 'm/s')]):
            pair_means = ds[f'pair_{quantity}_mean'].sel(polarization=pol)
            lag_means = ds[f'lag_{quantity}_mean'].sel(polarization=pol)
            axis.scatter(ds.antenna_lag, pair_means, color='0.45', label='Individual pair means')
            axis.plot(ds.lag, lag_means, 'o-', label='Mean of lag-averaged map')
            axis.axhline(0, color='0.7', linewidth=.8)
            axis.set_ylabel(unit)
            axis.grid(True, alpha=.25)
            axis.legend()
        axes[-1].set_xticks(ds.lag.values, [f'{int(lag)}\n({int(count)} pairs)'
                                            for lag, count in zip(ds.lag, ds.lag_pair_count)])
        axes[-1].set_xlabel('Antenna lag (channel-index difference)')
        figure.suptitle(f'ATI {pol}: average Doppler and radial velocity by lag')
        if destination is not None:
            figure.savefig(destination / f'ATI_lag_summary_{pol}.{plot_format}', dpi=150)
        if show:
            plt.show()
        plt.close(figure)


def plot_ati(ds, plot_path=None, *, show=False, plot_format='png', bounds=None):
    import matplotlib.pyplot as plt
    destination = Path(plot_path) if plot_path is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    bounds = map_plot_bounds(ds, bounds)
    lags = list(ds.lag.values) if 'lag_dc' in ds else [None]
    for pol in ds.polarization.values:
        for quantity, suffix, label, unit in (
                ('dc', 'doppler', 'ATI Doppler frequency', 'Hz'),
                ('radial_velocity', 'rvel', 'ATI radial velocity', 'm/s')):
            variable = f'lag_{quantity}' if lags[0] is not None else quantity
            limit = symmetric_limit(ds[variable].sel(polarization=pol))
            for lag in lags:
                values = ds[variable].sel(polarization=pol)
                lag_title = ''
                file_suffix = ''
                if lag is not None:
                    values = values.sel(lag=lag)
                    lag_title = f' lag {int(lag)} ({int(ds.lag_pair_count.sel(lag=lag))} pairs)'
                    file_suffix = f'_lag{int(lag)}' if len(lags) > 1 else ''
                figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
                mesh = axis.pcolormesh(ds.range, ds.azimuth, values,
                                       shading='auto', cmap='bwr', vmin=-limit, vmax=limit)
                figure.colorbar(mesh, ax=axis, label=f'{label} ({unit})')
                axis.set(xlabel='Ground range offset (m)', ylabel='Azimuth (m)',
                         title=f'{label} {pol}{lag_title} — mean {float(values.mean()):.2f} {unit}')
                set_map_bounds(axis, bounds)
                if destination is not None:
                    figure.savefig(destination / f'ATI_{suffix}_{pol}{pol}{file_suffix}.{plot_format}', dpi=150)
                if show:
                    plt.show()
                plt.close(figure)
    if 'lag_dc_mean' in ds:
        plot_lag_summary(ds, destination, show=show, plot_format=plot_format)


def plot_comparison(dc, ati, plot_path=None, *, show=False, channel=None, bounds=None):
    """Compare a DC channel with every ATI lag average in the ATI reference frame.

    Use ADPM-corrected DC/velocity maps when supplied; otherwise use stored-SLC
    phase frequency. ATI retains its InSAR flattening. No raster or channel
    averaging is applied to DC; channel defaults to the first selected.
    """
    import matplotlib.pyplot as plt
    dc = dc_plot_data(dc)
    channel_label = ''
    if 'channel' in dc.dims:
        channel = int(dc.channel.values[0]) if channel is None else channel
        dc = dc.sel(channel=channel, drop=True)
        channel_label = f' ch{channel}'
    bounds = map_plot_bounds(ati, bounds)
    selection = {axis: slice(*limits) for axis, limits in bounds.items()}
    dc, ati = dc.sel(**selection), ati.sel(**selection)
    if any(ds.sizes[axis] == 0 for ds in (dc, ati) for axis in bounds):
        raise ValueError('Shared DC/ATI domain has no output samples')
    destination = Path(plot_path) if plot_path is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    pols = [pol for pol in dc.polarization.values if pol in ati.polarization.values]
    if not pols:
        raise ValueError('DC and ATI have no common polarization labels')
    lags = list(ati.lag.values) if 'lag_dc' in ati else [None]
    for pol in pols:
        methods = list(dc.estimator.values)
        for lag in lags:
            figure, axes = plt.subplots(2, len(methods)+1, figsize=(5*(len(methods)+1), 8),
                                        sharex=True, sharey=True, constrained_layout=True,
                                        squeeze=False)
            lag_label = f' lag {int(lag)} average' if lag is not None else ''
            for row, (variable, unit) in enumerate((('dc', 'Hz'), ('radial_velocity', 'm/s'))):
                fields = [dc[variable].sel(polarization=pol, estimator=method) for method in methods]
                ati_values = ati[f'lag_{variable}'].sel(lag=lag) if lag is not None else ati[variable]
                fields.append(ati_values.sel(polarization=pol))
                # Shared scale across all ATI lags and this DC channel for each unit.
                all_ati = ati[f'lag_{variable}'].sel(polarization=pol) if lag is not None else fields[-1]
                limit = symmetric_limit(np.concatenate([field.values.ravel() for field in fields[:-1]]
                                                        + [all_ati.values.ravel()]))
                dc_label = 'Corrected DC' if dc.attrs.get('dc_plot_corrected', False) else 'DC'
                labels = [f'{dc_label}{channel_label} {m.upper()}' for m in methods]+[f'ATI{lag_label}']
                for column, (field, label) in enumerate(zip(fields, labels)):
                    axis = axes[row, column]
                    mesh = axis.pcolormesh(field.range, field.azimuth, field,
                                           shading='auto', cmap='bwr', vmin=-limit, vmax=limit)
                    axis.set(title=f'{label} ({unit})', xlabel='Ground range offset (m)',
                             ylabel='Azimuth (m)')
                    set_map_bounds(axis, bounds)
                figure.colorbar(mesh, ax=axes[row].tolist(), label=unit)
            figure.suptitle(f'{pol}: common window-centre coverage')
            suffix = f'_lag{int(lag)}' if len(lags) > 1 else ''
            if destination is not None:
                figure.savefig(destination / f'dc_ati_{pol}{suffix}.png', dpi=150)
            if show:
                plt.show()
            plt.close(figure)
