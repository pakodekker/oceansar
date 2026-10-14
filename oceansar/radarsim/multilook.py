"""Geometry-dependent multilook support, using the DC effective-width convention."""
import numpy as np


def multilook_window(pixel_spacing, window='hanning', *, resolution=None,
                     window_samples=None, image_shape=None):
    """Return sample counts and effective widths (azimuth, ground range).

    Width is sqrt(12 * weighted spatial variance + pixel_spacing**2).
    It describes the averaging kernel, not the combined SAR impulse response.
    Legacy sample counts are supported when no target resolution is supplied.
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
    """Human-readable report shared by InSAR and ATI."""
    support = np.asarray(counts) * spacing
    return (f'Multilooking ({window}; azimuth, ground range):\n'
            f'  Window samples: {int(counts[0])}, {int(counts[1])}\n'
            f'  Pixel spacing (m): {spacing[0]:.3f}, {spacing[1]:.3f}\n'
            f'  Window support (m): {support[0]:.3f}, {support[1]:.3f}\n'
            f'  Effective resolution (m): {effective[0]:.3f}, {effective[1]:.3f}\n')
