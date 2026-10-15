"""Same-polarization ATI estimates for all non-redundant antenna pairs."""
import numpy as np
import xarray as xr

from oceansar.constants import c


def uwphase(phasor):
    """Unwrap locally around the modal phase; no cross-lag ambiguity resolution."""
    phase = np.angle(phasor)
    finite = phase[np.isfinite(phase)]
    if not finite.size:
        return phase
    nbins = max(1, min(72, int(np.ceil(finite.size / 50))))
    histogram, edges = np.histogram(finite, bins=nbins, range=(-np.pi, np.pi))
    peak = histogram.argmax()
    reference = (edges[peak] + edges[peak+1]) / 2
    return np.angle(phasor * np.exp(-1j * reference)) + reference


def estimate_ati_pairs(coherences, coh_lut, baselines, v_orbit, f0,
                       polarizations, azimuth, ground_range):
    """Convert all same-polarization antenna pairs to Doppler and average by lag.

    Parameters
    ----------
    coherences : complex array
        Complex normalized InSAR coherences, shape (interferogram, azimuth,
        range). Their phases carry the input phase reference.
    coh_lut : integer array
        Lookup table with shape (channel, polarization, channel, polarization),
        mapping each non-redundant pair to its index in coherences.
    baselines : 1D array
        Along-track receive-antenna positions in metres, one per channel.
    v_orbit : float
        Positive orbital speed in m/s.
    f0 : float
        Positive carrier frequency in Hz.
    polarizations : sequence of str
        Polarization labels in lookup-table order.
    azimuth, ground_range : 1D array
        Output pixel-centre coordinates in metres, matching the final two
        dimensions of coherences.

    Returns
    -------
    xarray.Dataset
        pair_dc and pair_radial_velocity maps with axes (pair, polarization,
        azimuth, range), in Hz and m/s. Also includes pair_phase (radians),
        pair_coherence (dimensionless), signed baseline/time separations,
        lag averages, and finite-map spatial means and population standard
        deviations. dc/radial_velocity/phase aliases select pair 0-1.

    Raises
    ------
    ValueError
        Baselines, carrier, speed or lookup table are invalid, or a required
        pair is absent or has zero time separation.

    Notes
    -----
    Each pair uses time separation (baseline_j - baseline_i)/(2*v_orbit).
    Local modal phase unwrapping does not resolve ambiguities across lags.
    Velocity is -c * Doppler / (2*f0). Lag is the channel-index difference;
    lag maps equally average finite pair estimates after conversion to Hz.
    """
    baselines = np.asarray(baselines, dtype=float)
    if baselines.ndim != 1 or len(baselines) < 2 or not np.all(np.isfinite(baselines)):
        raise ValueError('ATI requires finite baselines for at least two channels')
    if not np.isfinite(v_orbit) or v_orbit <= 0 or not np.isfinite(f0) or f0 <= 0:
        raise ValueError('ATI requires positive finite v_orbit and f0')
    coherences = np.ma.asarray(coherences).filled(np.nan)
    lut = np.asarray(coh_lut)
    nch, npol = len(baselines), len(polarizations)
    if lut.shape != (nch, npol, nch, npol):
        raise ValueError('InSAR coherence LUT dimensions do not match channels/polarizations')
    pairs = [(i, j) for i in range(nch) for j in range(i+1, nch)]
    times = np.array([(baselines[j]-baselines[i])/(2*v_orbit) for i, j in pairs])
    if np.any(times == 0):
        raise ValueError('ATI pairs require nonzero along-track time separations')
    phases, selected_coherences = [], []
    for i, j in pairs:
        indices = [int(lut[i, p, j, p]) for p in range(npol)]
        if any(index < 0 or index >= len(coherences) for index in indices):
            raise ValueError(f'InSAR LUT is missing same-polarization pair {i}-{j}')
        selected = coherences[indices]
        selected = np.where(np.isfinite(selected) & (np.abs(selected) > 0), selected, np.nan+0j)
        phases.append(np.array([uwphase(values) for values in selected]))
        selected_coherences.append(np.abs(selected))
    phases = np.array(phases)
    frequency = phases / (2*np.pi*times[:, None, None, None])
    velocity = -c*frequency/(2*f0)
    dims = ('pair', 'polarization', 'azimuth', 'range')
    ds = xr.Dataset({
        'pair_phase': (dims, phases),
        'pair_dc': (dims, frequency),
        'pair_radial_velocity': (dims, velocity),
        'pair_coherence': (dims, np.array(selected_coherences)),
    }, coords={'pair': [f'{i}-{j}' for i, j in pairs],
               'polarization': polarizations, 'azimuth': azimuth, 'range': ground_range,
               'channel_start': ('pair', [i for i, j in pairs]),
               'channel_end': ('pair', [j for i, j in pairs]),
               'antenna_lag': ('pair', [j-i for i, j in pairs]),
               'baseline_m': ('pair', [baselines[j]-baselines[i] for i, j in pairs]),
               'time_separation_s': ('pair', times),
               'ambiguity_period_hz': ('pair', 1/np.abs(times))},
       attrs={'input_f0': float(f0), 'input_v_orbit': float(v_orbit),
              'reference_pair': '0-1', 'ati_time_separation_s': float(times[0]),
              'doppler_velocity_convention': 'v_r = -c * f_D / (2 * f0)',
              'lag_definition': 'channel_end - channel_start',
              'lag_averaging': 'equal-weight finite pair estimates per pixel after time-separation conversion',
              'ambiguity_resolution': 'local modal phase only; no cross-lag ambiguity resolution'})
    lag_maps = []
    for lag in range(1, nch):
        indices = np.flatnonzero(ds.antenna_lag.values == lag)
        selected = ds.isel(pair=indices)
        lag_maps.append(xr.Dataset({
            'lag_dc': selected.pair_dc.mean('pair', skipna=True),
            'lag_radial_velocity': selected.pair_radial_velocity.mean('pair', skipna=True),
            'lag_valid_pair_count': selected.pair_dc.count('pair'),
            'lag_pair_count': len(indices),
        }).expand_dims(lag=[lag]))
    ds = xr.merge([ds, xr.concat(lag_maps, dim='lag')])
    for prefix in ('pair', 'lag'):
        for quantity in ('dc', 'radial_velocity'):
            name = f'{prefix}_{quantity}'
            ds[f'{name}_mean'] = ds[name].mean(('azimuth', 'range'), skipna=True)
            ds[f'{name}_std'] = ds[name].std(('azimuth', 'range'), skipna=True)
    for name in ('dc', 'radial_velocity', 'phase'):
        ds[name] = ds[f'pair_{name}'].isel(pair=0, drop=True)
    ds['dc_mean'] = ds.pair_dc_mean.isel(pair=0, drop=True)
    ds['radial_velocity_mean'] = ds.pair_radial_velocity_mean.isel(pair=0, drop=True)
    ds['dc_std'] = ds.pair_dc_std.isel(pair=0, drop=True)
    ds['radial_velocity_std'] = ds.pair_radial_velocity_std.isel(pair=0, drop=True)
    for name in ds.data_vars:
        if 'radial_velocity' in name:
            ds[name].attrs['units'] = 'm/s'
        elif name == 'dc' or '_dc' in name or name in ('dc_mean', 'dc_std'):
            ds[name].attrs['units'] = 'Hz'
        elif 'phase' in name:
            ds[name].attrs['units'] = 'rad'
        if name.endswith('_std'):
            ds[name].attrs['statistic'] = 'Population spatial standard deviation of finite local map estimates (ddof=0)'
    for axis in ('azimuth', 'range'):
        ds[axis].attrs.update(units='m', long_name=f'{axis} window centre')
    ds.time_separation_s.attrs['units'] = 's'
    ds.baseline_m.attrs['units'] = 'm'
    ds.ambiguity_period_hz.attrs['units'] = 'Hz'
    return ds
