"""Madsen time-domain Doppler estimators for regularly sampled complex SLCs.

The observable is the input SLC's lag-correlation phase frequency, modulo
fs_az / lag. Conversion to acquisition Doppler requires a validated processing
model; this module does not infer that model from a nominal metadata centroid.

This module is shared unchanged between OceanSAR and s1d_ewv. It operates on
arrays and explicit sampling parameters, without project-specific I/O."""

import numpy as np
import xarray as xr


def _pair_mask(valid, segments, lag):
    ok = valid[lag:] & valid[:-lag]
    if segments is not None:
        # Every intervening edge must belong to the same contiguous segment.
        for j in range(lag):
            ok &= (segments[j:len(valid)-lag+j] ==
                   segments[j+1:len(valid)-lag+j+1])[:, None]
    return ok


def _pair_sums(slc, valid, segments, lag, weights=None):
    """Sufficient statistics, with no wrapping or crossing segment boundaries."""
    a, b = slc[lag:], slc[:-lag]
    ok = _pair_mask(valid, segments, lag)
    if weights is not None:
        ok &= weights > 0
    w = np.ones(np.count_nonzero(ok)) if weights is None else weights[ok]
    a, b = a[ok].astype(np.complex128), b[ok].astype(np.complex128)
    ia, qa = np.where(a.real >= 0, 1., -1.), np.where(a.imag >= 0, 1., -1.)
    ib, qb = np.where(b.real >= 0, 1., -1.), np.where(b.imag >= 0, 1., -1.)
    return np.array([len(a), np.sum(w * a * b.conj()),
                     np.sum(w * abs(a)**2), np.sum(w * abs(b)**2),
                     np.sum(w*ia*ib), np.sum(w*qa*qb),
                     np.sum(w*qa*ib), np.sum(w*ia*qb),
                     np.sum(w), np.sum(w*w)], dtype=np.complex128)


def _taper_weights(taper, shape):
    """Nonnegative weights on the lag-pair midpoint grid, not on SLC amplitude."""
    if taper is None:
        return None, 'boxcar'
    if isinstance(taper, (str, tuple)):
        from scipy.signal.windows import get_window
        weights = np.outer(get_window(taper, shape[0], fftbins=False),
                           get_window(taper, shape[1], fftbins=False))
        description = str(taper)
    elif callable(taper):
        weights = taper(tuple(shape))
        description = getattr(taper, '__name__', 'callable')
    else:
        weights = taper
        description = 'custom array'
    if np.iscomplexobj(weights):
        raise ValueError('taper weights must be real')
    weights = np.asarray(weights, float)
    if isinstance(taper, (str, tuple)):
        # Some mathematically nonnegative SciPy windows have roundoff at edges.
        weights = np.where((weights < 0) & (weights >= -64*np.finfo(float).eps), 0., weights)
    if weights.shape != tuple(shape):
        raise ValueError(f'taper weights must have lag-pair shape {tuple(shape)}')
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError('taper weights must be finite, nonnegative and not all zero')
    # Overall weight scale is immaterial; normalise to avoid overflow.
    return weights / weights.max(), description


def _spatial_moments(weights, shape, spacing):
    """Centre, second-moment width and covariance on the lag-pair grid."""
    if weights is None:
        return ((np.asarray(shape)-1)*spacing/2,
                np.asarray(shape)*spacing, 0.)
    total = weights.sum()
    if not total:
        return np.full(2, np.nan), np.full(2, np.nan), np.nan
    coordinates = [np.arange(n)*dx for n, dx in zip(shape, spacing)]
    marginals = [weights.sum(axis=1), weights.sum(axis=0)]
    centre = np.array([np.dot(w, x)/total for w, x in zip(marginals, coordinates)])
    variance = np.array([np.dot(w, (x-c)**2)/total
                         for w, x, c in zip(marginals, coordinates, centre)])
    covariance = np.sum(weights*(coordinates[0]-centre[0])[:, None]*
                        (coordinates[1]-centre[1])[None, :])/total
    return centre, np.sqrt(12*variance+spacing**2), covariance


def _resolve_window(taper, requested, spacing, image_shape, lag, window_samples,
                    samples_per_resolution=2.):
    """Size nominal support and derive the requested output sampling."""
    if window_samples is None:
        if taper is not None and not isinstance(taper, (str, tuple)):
            raise ValueError('Custom tapers require explicit window_samples (SLC support)')
        from scipy.signal.windows import get_window
        pair_shape = []
        for axis, (dx, target, limit) in enumerate(zip(spacing, requested, image_shape)):
            maximum = limit-lag if axis == 0 else limit
            target_pixels = max(target/dx, 2.)  # At least one-pixel half-width stride.

            def width(n):
                if taper is None or taper == 'boxcar':
                    return float(n)
                w = get_window(taper, n, fftbins=False)
                w = np.where((w < 0) & (w >= -64*np.finfo(float).eps), 0., w)
                if not np.all(np.isfinite(w)) or np.any(w < 0):
                    raise ValueError('Automatic sizing requires finite nonnegative taper weights')
                if not np.any(w > 0):
                    return 0.
                w = w/w.max()
                x = np.arange(n)
                centre = np.dot(w, x)/w.sum()
                return np.sqrt(12*np.dot(w, (x-centre)**2)/w.sum()+1.)

            if maximum < 2 or width(maximum) < target_pixels:
                raise ValueError('Requested effective resolution cannot fit the image with this taper')
            # Named symmetric nonnegative windows have increasing second-moment
            # widths. Find the smallest admissible count, not an even-pixel size.
            lo, hi = 2, maximum
            while lo < hi:
                mid = (lo+hi)//2
                if width(mid) >= target_pixels:
                    hi = mid
                else:
                    lo = mid+1
            pair_shape.append(lo)
        window = np.array(pair_shape)+np.array([lag, 0])
    else:
        supplied = np.asarray(window_samples)
        if supplied.shape != (2,) or not np.issubdtype(supplied.dtype, np.integer) or np.issubdtype(supplied.dtype, np.bool_):
            raise ValueError('window_samples must be two integers (azimuth, range)')
        window = supplied.astype(int)
    if window[0] <= lag or window[1] < 1 or np.any(window > image_shape):
        raise ValueError('window_samples must fit the image and azimuth support must exceed lag')
    pair_shape = tuple(window-np.array([lag, 0]))
    weights, description = _taper_weights(taper, pair_shape)
    centre, effective, covariance = _spatial_moments(weights, pair_shape, spacing)
    if np.any(effective < requested*(1-1e-12)):
        raise ValueError('Explicit window support does not meet requested effective resolution')
    stride = np.maximum(1, np.floor(np.nextafter(
        effective/(samples_per_resolution*spacing), np.inf)).astype(int))
    return window, weights, description, centre, effective, covariance, stride


def _estimates(stats, methods, fs_az, lag, min_pairs, min_correlation):
    count = int(stats[0].real)
    weight_sum = stats[8].real
    if not weight_sum:
        return np.full(len(methods), np.nan), np.full(len(methods), np.nan+0j), np.nan
    rho = np.sin(np.pi/2 * stats[4:8].real/weight_sum)
    correlations = {'cde': stats[1]/weight_sum,
                    'sde': .5*(rho[0]+rho[1]) + .5j*(rho[2]-rho[3])}
    denominator = np.sqrt(stats[2].real * stats[3].real)
    coherence = abs(stats[1])/denominator if denominator else np.nan
    corr = np.array([correlations[m] for m in methods])
    strength = np.array([coherence if m == 'cde' else abs(correlations[m]) for m in methods])
    phase = (np.angle(corr)+np.pi) % (2*np.pi)-np.pi
    frequency = phase*fs_az/(2*np.pi*lag)
    frequency[(count < min_pairs) | (~np.isfinite(strength)) |
              (strength <= min_correlation)] = np.nan
    return frequency, corr, coherence


def estimate_dc(slc, *, fs_az, pixel_spacing, resolution, methods=('cde', 'sde'),
                lag=1, valid_mask=None, segment_ids=None, nodata=None,
                min_pairs=1, min_correlation=0., origin=(0., 0.), taper=None,
                window_samples=None, samples_per_resolution=2.):
    """Return local DC fields and pooled whole-image estimates as an xr.Dataset.

    Parameters
    ----------
    slc : complex 2D array
        Axes are (azimuth, range); no basebanding or filtering is applied.
    fs_az : float
        Sampling frequency of the supplied azimuth grid, Hz (not necessarily PRF).
    pixel_spacing, resolution : scalar or pair of floats
        Spatial sampling and requested second-moment effective resolution in
        metres, (azimuth, range). Derive integer support from the actual taper
        so sqrt(12*weighted_variance + pixel_spacing**2) meets the request.
        Azimuth SLC support includes lag extra rows. Output stride rounds down
        to whole pixels at effective resolution / samples_per_resolution,
        with a minimum stride of one input pixel. Only full
        windows are returned. This is not a strict band-limit guarantee.
    methods : sequence of {'cde', 'sde'}
        CDE uses complex lag products. SDE inverts each of the four component
        sign correlations with the arcsine law before reconstructing phase.
        SDE's statistical model assumes zero-mean proper complex Gaussian data.
    lag : int, optional
        Positive azimuth separation in input samples; default 1.
    valid_mask : bool array, optional
        True for valid pixels. Nonfinite samples are always excluded.
    segment_ids : 1D array, optional
        Azimuth continuity labels (e.g. bursts). No pair crosses a label change.
        Mask any gaps explicitly using valid_mask.
    nodata : scalar, optional
        Explicit invalid value; zero is otherwise a valid signal sample.
    min_pairs, min_correlation : int, float
        Quality thresholds; strength is normalised CDE coherence or SDE |rho|.
    origin : pair
        Position in metres of the first input pixel centre.
    taper : str, tuple, callable or 2D array, optional
        Local spatial averaging weights. None uses a rectangular average.
        A scipy window specification (e.g. 'hann', 'hamming', ('tukey', 0.5))
        creates a separable symmetric taper on the lag-pair midpoint grid,
        whose shape is (azimuth_window_samples-lag, range_window_samples).
        A callable accepts that shape and returns weights; a custom array must
        match it. Weights must be finite, nonnegative and not all zero.
        Weight the complex products and each sign correlation directly, then
        normalise by the sum of weights over valid pairs. Do not taper the
        SLC before taking component signs. Support expands for narrower tapers;
        for Hann, approximately 1.597 times the requested effective resolution.
    window_samples : pair of integers, optional
        Explicit SLC support (azimuth, range), required for callable/array tapers
        whose resizing rule is unknown. Its nominal effective resolution must
        meet the request. The stride still follows the effective resolution.

    samples_per_resolution : scalar or pair of floats
        Output samples per effective resolution (azimuth, range), at least 1.
        Default 2; increase to 4 or 8 for denser maps at the same resolution.
        Sampling is limited by input pixels; realised spacing is reported.

    Returns
    -------
    xarray.Dataset
        Local dc maps in Hz with dimensions (estimator, azimuth, range),
        complex lag correlations, normalized coherence, valid-pair counts,
        effective pair counts and pooled dc_mean values. Coordinates are
        window centres in metres. dc_mean counts each whole-image valid pair
        once, independently of local overlap. Attributes report the window
        support, effective widths, input spacing and realized output spacing.

    Notes
    -----
    Forward products are s[n+lag,m] * conj(s[n,m]), giving positive frequency
    for exp(+j*2*pi*f*n/fs_az). Frequencies lie in [-fs_az/(2*lag), fs_az/(2*lag)).
    dc_mean pools valid pairs from the entire input, counting each pair once;
    it remains an unweighted full-image estimate when a local taper is used,
    and is not an arithmetic average of the overlapping dc windows.
    effective_pairs = (sum w)**2 / sum(w**2) describes weight concentration;
    it is not the number of statistically independent samples.
    """
    slc = np.asarray(slc)
    spacing = np.broadcast_to(np.asarray(pixel_spacing, float), (2,))
    requested = np.broadcast_to(np.asarray(resolution, float), (2,))
    origin = np.broadcast_to(np.asarray(origin, float), (2,))
    sampling = np.broadcast_to(np.asarray(samples_per_resolution, float), (2,))
    if np.any(~np.isfinite(sampling)) or np.any(sampling < 1):
        raise ValueError("samples_per_resolution must be finite and at least 1")
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    if slc.ndim != 2 or not np.iscomplexobj(slc):
        raise ValueError('slc must be a complex array with axes (azimuth, range)')
    if not np.isfinite(fs_az) or fs_az <= 0:
        raise ValueError('fs_az must be finite and positive')
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0) or np.any(~np.isfinite(requested)) or np.any(requested <= 0):
        raise ValueError('pixel_spacing and resolution must be finite and positive')
    if not np.all(np.isfinite(origin)):
        raise ValueError('origin must be finite')
    if not methods or len(set(methods)) != len(methods) or any(m not in ('cde', 'sde') for m in methods):
        raise ValueError('methods must contain cde and/or sde without duplicates')
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError('lag must be a positive integer')
    if not isinstance(min_pairs, (int, np.integer)) or min_pairs < 1 or not np.isfinite(min_correlation) or min_correlation < 0:
        raise ValueError('min_pairs must be positive integer; min_correlation finite and nonnegative')
    window, weights, taper_description, pair_centre, effective, covariance, stride = _resolve_window(
        taper, requested, spacing, slc.shape, lag, window_samples, sampling)
    valid = np.isfinite(slc)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != slc.shape:
            raise ValueError('valid_mask must match slc shape')
        valid &= mask
    if nodata is not None:
        valid &= slc != nodata
    segments = None if segment_ids is None else np.asarray(segment_ids)
    if segments is not None and segments.shape != (len(slc),):
        raise ValueError('segment_ids must have one label per azimuth row')
    starts = [np.arange(0, size-win+1, step) for size, win, step in zip(slc.shape, window, stride)]
    shape = (len(methods), len(starts[0]), len(starts[1]))
    dc, corr = np.empty(shape), np.empty(shape, complex)
    counts, coherence = np.empty(shape[1:], np.int64), np.empty(shape[1:])
    weight_sums, effective_pairs = np.empty(shape[1:]), np.empty(shape[1:])
    valid_resolution, centre_shift = np.empty((2, *shape[1:])), np.empty((2, *shape[1:]))
    valid_covariance = np.empty(shape[1:])
    for i, y in enumerate(starts[0]):
        sy = slice(y, y+window[0])
        for j, x in enumerate(starts[1]):
            sx = slice(x, x+window[1])
            stats = _pair_sums(slc[sy, sx], valid[sy, sx],
                               None if segments is None else segments[sy], lag, weights)
            dc[:, i, j], corr[:, i, j], coherence[i, j] = _estimates(
                stats, methods, fs_az, lag, min_pairs, min_correlation)
            counts[i, j] = int(stats[0].real)
            weight_sums[i, j] = stats[8].real
            effective_pairs[i, j] = stats[8].real**2/stats[9].real if stats[9].real else 0.
            support = _pair_mask(valid[sy, sx], None if segments is None else segments[sy], lag)
            if support.all():
                valid_resolution[:, i, j] = effective
                centre_shift[:, i, j] = 0.
                valid_covariance[i, j] = covariance
            else:
                surviving = support.astype(float) if weights is None else weights*support
                actual_centre, actual_resolution, actual_covariance = _spatial_moments(
                    surviving, support.shape, spacing)
                valid_resolution[:, i, j] = actual_resolution
                centre_shift[:, i, j] = actual_centre-pair_centre
                valid_covariance[i, j] = actual_covariance
    # Bound temporary memory; include a lag halo, assigning each earlier row once.
    total = np.zeros(10, complex)
    for y in range(0, len(slc)-lag, 256):
        sy = slice(y, min(y+256, len(slc)-lag)+lag)
        total += _pair_sums(slc[sy], valid[sy], None if segments is None else segments[sy], lag)
    mean, mean_corr, mean_coherence = _estimates(total, methods, fs_az, lag, min_pairs, min_correlation)
    dims = ('estimator', 'azimuth', 'range')
    ds = xr.Dataset({
        'dc': (dims, dc), 'correlation_real': (dims, corr.real),
        'correlation_imag': (dims, corr.imag),
        'valid_pairs': (dims[1:], counts), 'cde_coherence': (dims[1:], coherence),
        'weight_sum': (dims[1:], weight_sums), 'effective_pairs': (dims[1:], effective_pairs),
        'valid_resolution_azimuth': (dims[1:], valid_resolution[0]),
        'valid_resolution_range': (dims[1:], valid_resolution[1]),
        'averaging_centre_shift_azimuth': (dims[1:], centre_shift[0]),
        'averaging_centre_shift_range': (dims[1:], centre_shift[1]),
        'valid_spatial_covariance': (dims[1:], valid_covariance),
        'dc_mean': ('estimator', mean),
        'mean_correlation_real': ('estimator', mean_corr.real),
        'mean_correlation_imag': ('estimator', mean_corr.imag),
        'mean_valid_pairs': int(total[0].real), 'mean_cde_coherence': mean_coherence,
    }, coords={'estimator': list(methods),
               'azimuth': origin[0]+starts[0]*spacing[0]+lag*spacing[0]/2+pair_centre[0],
               'range': origin[1]+starts[1]*spacing[1]+pair_centre[1]},
       attrs={'observable': 'SLC lag-correlation phase frequency; acquisition centroid uncorrected',
              'fs_az_hz': float(fs_az), 'lag': int(lag),
              'requested_resolution_m': requested.tolist(), 'pixel_spacing_m': spacing.tolist(),
              'window_samples': window.tolist(), 'window_support_m': (window*spacing).tolist(),
              'resolution_m': effective.tolist(), 'effective_resolution_m': effective.tolist(),
              'resolution_definition': 'sqrt(12*nominal weighted spatial variance + pixel spacing squared)',
              'nominal_spatial_covariance_m2': float(covariance),
              'window_sizing': 'automatic minimal support' if window_samples is None else 'explicit support',
              'samples_per_resolution': sampling.tolist(),
              'realised_samples_per_resolution': (effective/(stride*spacing)).tolist(),
              'output_stride_samples': stride.tolist(), 'output_spacing_m': (stride*spacing).tolist(),
              'support_overlap_fraction': (1-stride/window).tolist(),
              'min_pairs': int(min_pairs), 'min_correlation': float(min_correlation),
              'phase_interval': '[-pi, pi)', 'mean_definition': 'pooled unique valid image pairs',
              'local_taper': taper_description, 'mean_taper': 'boxcar',
              'weight_definition': 'direct lag-pair weights on midpoint grid, maximum normalised to one',
              'continuity': 'explicit segment labels' if segments is not None else 'assumed contiguous input'})
    for name in ('dc', 'dc_mean'):
        ds[name].attrs.update(units='Hz', long_name='SLC correlation-phase Doppler frequency')
    for name in ('azimuth', 'range'):
        ds[name].attrs.update(units='m', long_name=f'{name} window centre')
    ds.effective_pairs.attrs['long_name'] = 'Effective pair count from weights, not statistical independence'
    ds.valid_pairs.attrs['long_name'] = 'Valid lag pairs with positive weight'
    for name in ('valid_resolution_azimuth', 'valid_resolution_range',
                 'averaging_centre_shift_azimuth', 'averaging_centre_shift_range'):
        ds[name].attrs['units'] = 'm'
    ds.valid_spatial_covariance.attrs['units'] = 'm2'
    return ds

