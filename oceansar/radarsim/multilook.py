"""Geometry-dependent multilook support, using the DC effective-width convention."""
import numpy as np


def multilook_window(pixel_spacing, window='hanning', *, resolution=None,
                     window_samples=None, image_shape=None):
    """Choose multilook sample counts and report their effective spatial widths.

    Parameters
    ----------
    pixel_spacing : pair of float
        Input sample spacing in metres (azimuth, ground range).
    window : {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}, optional
        Separable averaging-window shape; default 'hanning'.
    resolution : float or pair of float, optional
        Target effective widths in metres (azimuth, ground range). Select the
        smallest integer support that meets each requested width.
    window_samples : pair of int, optional
        Explicit support (azimuth, range), used when resolution is None.
    image_shape : pair of int, optional
        Maximum support counts (azimuth, range). Without a shape, each axis
        is limited to one million samples.

    Returns
    -------
    counts : numpy.ndarray
        Integer averaging support counts, shape (2,), in axis order.
    effective : numpy.ndarray
        Realized widths in metres, shape (2,), computed as
        sqrt(12 * weighted_spatial_variance + pixel_spacing**2). These describe
        the averaging kernel, not the combined SAR impulse response.

    Raises
    ------
    ValueError
        Controls are invalid or the requested support cannot fit the image.
    """
    spacing = np.asarray(pixel_spacing, dtype=float)
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('pixel_spacing must contain two positive finite values')
    if window not in ('flat', 'hanning', 'hamming', 'bartlett', 'blackman'):
        raise ValueError('Unsupported multilooking window')

    def width(n, dx):
        # utils.smooth leaves non-flat windows of fewer than three samples unchanged.
        if window != 'flat' and n < 3:
            return dx
        weights = np.ones(n) if window == 'flat' else getattr(np, window)(n)
        weights = np.maximum(weights, 0)
        if weights.sum() == 0:
            return dx
        x = np.arange(n)
        centre = np.dot(weights, x) / weights.sum()
        return dx * np.sqrt(12 * np.dot(weights, (x-centre)**2) / weights.sum() + 1)

    limits = np.asarray(image_shape if image_shape is not None else (1000000, 1000000))
    if limits.shape != (2,) or np.any(limits < 1):
        raise ValueError('image_shape must contain two positive sizes')
    if resolution is not None:
        target = np.broadcast_to(np.asarray(resolution, dtype=float), (2,))
        if not np.all(np.isfinite(target)) or np.any(target <= 0):
            raise ValueError('resolution must be positive and finite')
        counts = []
        for dx, requested, limit in zip(spacing, target, limits):
            lo, hi = 1, int(limit)
            if width(hi, dx) < requested:
                raise ValueError('Requested multilook resolution cannot fit the image')
            while lo < hi:
                mid = (lo + hi) // 2
                if width(mid, dx) >= requested:
                    hi = mid
                else:
                    lo = mid + 1
            counts.append(lo)
        counts = np.asarray(counts)
    else:
        supplied = np.asarray(window_samples)
        if (supplied.shape != (2,) or not np.all(np.isfinite(supplied))
                or np.any(supplied < 1) or np.any(supplied != supplied.astype(int))):
            raise ValueError('window_samples must contain two positive integers')
        counts = supplied.astype(int)
        if np.any(counts > limits):
            raise ValueError('Multilook window must fit the image')
    effective = np.array([width(n, dx) for n, dx in zip(counts, spacing)])
    return counts, effective


def multilook_report(counts, spacing, effective, window):
    """Format the realized multilook support for logs and statistics reports.

    Parameters
    ----------
    counts : pair of int
        Window sample counts (azimuth, ground range).
    spacing : pair of float
        Input sample spacing in metres, in the same axis order.
    effective : pair of float
        Effective second-moment averaging widths in metres.
    window : str
        Averaging-window name included in the report.

    Returns
    -------
    str
        Multiline report of sample counts, input spacing, physical support
        (counts * spacing) and effective resolution, ending with a newline.
    """
    support = np.asarray(counts) * spacing
    return (f'Multilooking ({window}; azimuth, ground range):\n'
            f'  Window samples: {int(counts[0])}, {int(counts[1])}\n'
            f'  Pixel spacing (m): {spacing[0]:.3f}, {spacing[1]:.3f}\n'
            f'  Window support (m): {support[0]:.3f}, {support[1]:.3f}\n'
            f'  Effective resolution (m): {effective[0]:.3f}, {effective[1]:.3f}\n')
