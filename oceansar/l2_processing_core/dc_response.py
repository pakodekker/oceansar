"""Finite-lag Doppler response calibration from an explicit spectral model.

Implements dc_burst_sensitivity_adpm_draft.rst: the shift derivative moves
the antenna spectrum while the burst, processing and equalization stay fixed.
This module does not infer a burst model from a nominal Doppler centroid.
"""

import numpy as np
import xarray as xr


def doppler_response(frequency, power, derivative, *, fs_az, lag=1):
    """Integrate reference correlation and its derivative on a frequency grid.

    Parameters
    ----------
    frequency : array-like
        Strictly increasing frequencies in Hz on the final axis. A 1D grid
        or grids broadcastable to power are accepted.
    power : array-like
        Finite nonnegative reference power spectra. Leading axes identify
        independent local models; the final axis is frequency.
    derivative : array-like
        Signed derivative of power with respect to the physical Doppler
        shift, in power/Hz. Include only terms that move under that shift.
    fs_az : float
        Actual SLC azimuth sampling frequency in Hz.
    lag : int, optional
        Positive lag in input azimuth samples, default 1.

    Returns
    -------
    dict of numpy.ndarray
        Frequency-integrated power, power_derivative, complex correlation
        and correlation_derivative; reference_phase in Hz, dimensionless
        sensitivity and correlation_strength. Shapes equal the leading
        power dimensions. Zero-power/correlation models yield NaN response.
        sensitivity = imag(correlation_derivative/correlation) / q, with
        q = 2*pi*lag/fs_az. No sensitivity clipping is performed.
    """
    if not np.isfinite(fs_az) or fs_az <= 0:
        raise ValueError('fs_az must be positive and finite')
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError('lag must be a positive integer')
    f, p, dp = np.broadcast_arrays(np.asarray(frequency, float),
                                   np.asarray(power, float), np.asarray(derivative, float))
    if (p.ndim < 1 or p.shape[-1] < 3 or not np.all(np.isfinite(f))
            or np.any(np.diff(f, axis=-1) <= 0) or not np.all(np.isfinite(p))
            or np.any(p < 0) or not np.all(np.isfinite(dp))):
        raise ValueError('Supply increasing finite frequency grids, nonnegative power and finite derivatives')
    q = 2*np.pi*lag/fs_az
    kernel = np.exp(1j*q*f)
    # np.trapezoid is unavailable in NumPy 1.x, still supported by the hosts.
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    energy = integrate(p, f, axis=-1)
    denergy = integrate(dp, f, axis=-1)
    correlation = integrate(p*kernel, f, axis=-1)
    dcorrelation = integrate(dp*kernel, f, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        strength = abs(correlation)/energy
        sensitivity = np.imag(dcorrelation/correlation)/q
    usable = (energy > 0) & (abs(correlation) > np.finfo(float).eps*energy)
    return dict(power=energy, power_derivative=denergy, correlation=correlation,
                correlation_derivative=dcorrelation,
                reference_phase=np.where(usable, np.angle(correlation)/q, np.nan),
                sensitivity=np.where(usable, sensitivity, np.nan),
                correlation_strength=np.where(usable, strength, np.nan))


def pool_doppler_response(response, weights, *, fs_az, lag=1,
                          methods=('cde', 'sde')):
    """Calibrate the actual CDE/SDE mixtures across a spatial averaging window.

    Parameters
    ----------
    response : dict of array
        One-dimensional local spectral integrals from doppler_response.
        The samples describe spatial cells within a single estimator window.
    weights : 1D array
        Finite nonnegative lag-pair weights, matching the local models.
        Zero-weight/zero-power cells do not contribute.
    fs_az : float
        Actual SLC azimuth sampling frequency in Hz.
    lag : int, optional
        Correlation lag in samples, default 1.
    methods : sequence of {'cde', 'sde'}, optional
        Output estimator order. SDE pools Gaussian sign correlations before
        the arcsine-law inversion, rather than averaging phase responses.

    Returns
    -------
    dict of numpy.ndarray
        reference_phase (Hz), sensitivity (dimensionless), and
        correlation_strength, one entry per requested estimator. This is
        a proper, zero-mean complex Gaussian population response model.
    """
    if (not np.isfinite(fs_az) or fs_az <= 0 or isinstance(lag, bool)
            or not isinstance(lag, (int, np.integer)) or lag < 1):
        raise ValueError('fs_az must be positive finite and lag a positive integer')
    w = np.asarray(weights, float)
    energy = np.asarray(response['power'])
    if (w.ndim != 1 or energy.shape != w.shape or not np.all(np.isfinite(w))
            or np.any(w < 0) or not np.any(w > 0)):
        raise ValueError('weights must match the 1D response, be nonnegative and have positive total')
    usable = (w > 0) & (energy > 0)
    if not usable.any():
        return {name: np.full(len(methods), np.nan)
                for name in ('reference_phase', 'sensitivity', 'correlation_strength')}
    w = w[usable]/w[usable].sum()
    energy = energy[usable]
    de = np.asarray(response['power_derivative'])[usable]
    r = np.asarray(response['correlation'])[usable]
    dr = np.asarray(response['correlation_derivative'])[usable]
    rho = r/energy
    drho = dr/energy-r*de/energy**2
    correlations, derivatives, strengths = [], [], []
    for method in methods:
        if method == 'cde':
            value, slope = np.dot(w, r), np.dot(w, dr)
            strength = abs(value)/np.dot(w, energy)
        elif method == 'sde':
            # E[sign(x) sign(y)] = 2/pi * asin(rho). Pool the component
            # expectations before inversion, exactly as the SDE estimator.
            real = np.clip(rho.real, -1+1e-12, 1-1e-12)
            imag = np.clip(rho.imag, -1+1e-12, 1-1e-12)
            a, b = np.dot(w, np.arcsin(real)), np.dot(w, np.arcsin(imag))
            value = np.sin(a) + 1j*np.sin(b)
            slope = (np.cos(a)*np.dot(w, drho.real/np.sqrt(1-real**2))
                     + 1j*np.cos(b)*np.dot(w, drho.imag/np.sqrt(1-imag**2)))
            strength = abs(value)
        else:
            raise ValueError('methods must contain cde or sde')
        correlations.append(value)
        derivatives.append(slope)
        strengths.append(strength)
    r, dr = np.asarray(correlations), np.asarray(derivatives)
    q = 2*np.pi*lag/fs_az
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.imag(dr/r)/q
    return dict(reference_phase=np.angle(r)/q, sensitivity=gain,
                correlation_strength=np.asarray(strengths))


def correct_dc_response(ds, reference_phase, sensitivity, model_strength, *,
                        model_valid=True, min_sensitivity=.01,
                        min_correlation=.05, max_shift_hz=50.):
    """Add a local Doppler-shift correction without changing raw DC products.

    Parameters
    ----------
    ds : xarray.Dataset
        estimate_dc result with dc in Hz and fs_az_hz/lag attributes.
    reference_phase, sensitivity, model_strength : xarray.DataArray
        Baseline phase frequency in Hz, dimensionless shift sensitivity g,
        and reference correlation strength, aligned with ds.dc.
    model_valid : bool or xarray.DataArray, optional
        Model-domain validity, including the supplied linearization interval.
    min_sensitivity, min_correlation : float, optional
        Minimum absolute sensitivity and model correlation strength. Defaults
        are .01 and .05; rejected corrections remain NaN, never clipped.
    max_shift_hz : float, optional
        Symmetric local shift interval around zero in Hz, default 50.
        The nearest phase branch is used; no absolute ambiguity is resolved.

    Returns
    -------
    xarray.Dataset
        Copy with dc_reference_phase, dc_sensitivity, dc_scale_factor=1/g,
        dc_model_correlation, dc_correction_valid/reason, and
        dc_shift_corrected=(dc-baseline)/g on the nearest phase branch.
        Corrected mean/std are spatial statistics of the corrected maps.
        Raw dc, dc_mean and pooled-correlation semantics remain unchanged.
    """
    if (not np.isfinite(min_sensitivity) or min_sensitivity <= 0
            or not np.isfinite(min_correlation) or not 0 <= min_correlation <= 1
            or not np.isfinite(max_shift_hz) or max_shift_hz <= 0):
        raise ValueError('Invalid response thresholds or max_shift_hz')
    q = 2*np.pi*ds.attrs['lag']/ds.attrs['fs_az_hz']
    result = ds.copy()
    baseline, gain, strength, measured = (
        value.transpose(*ds.dc.dims)
        for value in xr.broadcast(reference_phase, sensitivity, model_strength, ds.dc))
    domain = xr.broadcast(xr.DataArray(model_valid) if np.isscalar(model_valid) else model_valid, measured)[0]
    domain = domain.transpose(*ds.dc.dims).astype(bool)
    domain = domain & (abs(gain)*max_shift_hz < ds.attrs['fs_az_hz']/(2*ds.attrs['lag']))
    reason = xr.zeros_like(measured, dtype=np.uint8)
    reason = reason | xr.where(~np.isfinite(baseline) | ~np.isfinite(gain) | ~np.isfinite(strength), 1, 0).astype('uint8')
    reason = reason | xr.where(strength < min_correlation, 2, 0).astype('uint8')
    reason = reason | xr.where(abs(gain) < min_sensitivity, 4, 0).astype('uint8')
    reason = reason | xr.where(~np.isfinite(measured), 8, 0).astype('uint8')
    reason = reason | xr.where(~domain, 32, 0).astype('uint8')
    residual = np.angle(np.exp(1j*q*(measured-baseline)))/q
    with np.errstate(divide='ignore', invalid='ignore'):
        shift = residual/gain
        scale = 1/gain
    reason = reason | xr.where(abs(shift) > max_shift_hz, 16, 0).astype('uint8')
    result['dc_reference_phase'] = baseline
    result['dc_sensitivity'] = gain
    result['dc_model_correlation'] = strength
    result['dc_scale_factor'] = scale.where((reason & np.uint8(1 | 2 | 4 | 32)) == 0)
    result['dc_correction_reason'] = reason
    result['dc_correction_valid'] = (reason == 0).astype('uint8')
    result['dc_shift_corrected'] = shift.where(reason == 0)
    result['dc_shift_corrected_mean'] = result.dc_shift_corrected.mean(('azimuth', 'range'), skipna=True)
    result['dc_shift_corrected_std'] = result.dc_shift_corrected.std(('azimuth', 'range'), skipna=True)
    for name in ('dc_reference_phase', 'dc_shift_corrected', 'dc_shift_corrected_mean', 'dc_shift_corrected_std'):
        result[name].attrs['units'] = 'Hz'
    for name in ('dc_sensitivity', 'dc_scale_factor', 'dc_model_correlation'):
        result[name].attrs['units'] = '1'
    result.dc_correction_reason.attrs.update(
        flag_masks=np.array([1, 2, 4, 8, 16, 32], dtype='uint8'),
        flag_meanings='invalid_model weak_model_correlation low_sensitivity invalid_measurement outside_shift_interval outside_model_domain')
    result.attrs.update(dc_response_correction='ADPM finite-lag local response; reference shift zero',
                        dc_correction_min_sensitivity=float(min_sensitivity),
                        dc_correction_min_correlation=float(min_correlation),
                        dc_correction_max_shift_hz=float(max_shift_hz),
                        dc_correction_phase_branch='nearest reference branch; absolute ambiguity unresolved',
                        dc_corrected_mean_definition='Arithmetic mean of finite corrected local maps; pooled dc_mean remains raw')
    return result
