"""Cross-spectra of detected ocean SAR intensities on a regular spatial grid.

The shared kernels accept intensities, never detect SLCs implicitly, and use
F_forward * conj(F_backward). Wavenumbers are in cycles/m, not radians/m.
DRAMA supplies the existing smoothing and FFT-size utilities.
"""

import numpy as np
from scipy import fft
from drama import utils as drutils


def _spacing(pixel_spacing):
    spacing = np.broadcast_to(np.asarray(pixel_spacing, dtype=float), (2,))
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('pixel_spacing must be positive finite metres (azimuth, range)')
    return spacing


def _intensities(int_fwd, int_bwd):
    images = tuple(np.asarray(image) for image in (int_fwd, int_bwd))
    if (images[0].ndim != 2 or images[0].shape != images[1].shape
            or min(images[0].shape) < 2):
        raise ValueError('Intensity images must have equal 2D shapes with at least two samples per axis')
    for image in images:
        if np.iscomplexobj(image) or not np.all(np.isfinite(image)) or np.any(image < 0):
            raise ValueError('Supply finite nonnegative real intensities, not complex SLCs')
    return images


def outlier_removal(int_img, thresh=3.0, lp_scale=2e3,
                    pixel_spacing=5.0, second=False):
    """Replace bright intensity outliers using a smoothed local reference.

    Parameters
    ----------
    int_img : 2D real array
        Detected intensity image, ordered (azimuth, range).
    thresh : float, optional
        Relative excess above the local mean that triggers replacement;
        default 3 means values greater than four times that mean.
    lp_scale : float, optional
        Low-pass window support in metres, default 2000.
    pixel_spacing : float, optional
        Representative input spacing in metres, default 5. The square
        smoothing window has max(1, int(lp_scale/pixel_spacing)) samples.
        Cross-spectrum kernels use the geometric mean of the two spacings.
    second : bool, optional
        Repeat the rejection using a reference smoothed after the first pass.

    Returns
    -------
    numpy.ndarray
        New intensity array with the input shape. Rejected samples are
        replaced by 1e-6 in input intensity units; the input is unchanged.
    """
    if not np.isfinite(pixel_spacing) or pixel_spacing <= 0:
        raise ValueError('pixel_spacing must be positive and finite')
    if not np.isfinite(lp_scale) or lp_scale <= 0 or not np.isfinite(thresh) or thresh < 0:
        raise ValueError('lp_scale must be positive and thresh nonnegative, both finite')
    window_len = max(1, int(lp_scale / pixel_spacing))
    result = np.asarray(int_img)
    for _ in range(2 if second else 1):
        reference = drutils.smooth(result, window_len)
        denominator = np.where(reference == 0, 1e-6, reference)
        result = np.where((result-reference)/denominator > thresh, 1e-6, result)
    return result


def welch_xspec(int_fwd, int_bwd, *, pixel_spacing, remove_outliers=True,
                window_size=(2e3, 2e3), lp_scale=1e3, thresh=100.0,
                tile_normalization=True, tile_outlier_threshold=3,
                win_func=None, verbose=False):
    """Estimate an intensity cross-spectrum using overlapping spatial tiles.

    Parameters
    ----------
    int_fwd, int_bwd : 2D real array
        Finite nonnegative intensity images with equal shape (azimuth,
        ground range). Convert complex SLCs to abs(SLC)**2 before calling.
    pixel_spacing : float or pair of float
        Input sample spacing in metres (azimuth, ground range).
    remove_outliers : bool, optional
        Apply outlier_removal to both images before tiling; default True.
    window_size : float or pair of float, optional
        Requested tile support in metres (azimuth, ground range), default
        (2000, 2000). Integer counts round to FFT sizes with prime factors
        at most 3, bounded by the image. Strides are max(1, count//2).
        Only full tiles are used; trailing incomplete tiles are omitted.
    lp_scale, thresh : float, optional
        Outlier reference support in metres and relative-excess threshold,
        passed to outlier_removal. Defaults are 1000 and 100.
    tile_normalization : bool, optional
        Subtract each tile's mean, then divide by that mean when True
        (default), or by the corresponding full-image mean when False.
    tile_outlier_threshold : float, optional
        Reject tiles whose mean brightness exceeds the mean tile brightness
        plus this many population standard deviations; default 3. Brightness
        is the average of the two original tile means, before centering.
    win_func : callable, optional
        Nonnegative 1D taper callable taking a sample count. Default is
        numpy.hamming; form the 2D window as its separable outer product.
    verbose : bool, optional
        Print the number of tiles along range and azimuth.

    Returns
    -------
    kx, ky : numpy.ndarray
        Shifted range and azimuth wavenumber axes in cycles/m, respectively.
    average : numpy.ndarray
        Complex64 cross-power averaged over accepted tiles, shape (ky, kx).
        Uses FFT(fwd) * conj(FFT(bwd)), divided by sum(window**2). This
        preserves the existing discrete, relative-intensity normalization;
        it is not a spectral density per unit physical wavenumber area.
    tiles : numpy.ndarray
        Complex64 individual spectra, shape (azimuth_tile, range_tile,
        ky, kx), including bright tiles rejected from average. Tiles with
        a zero normalization mean are NaN and excluded from the average.

    Raises
    ------
    ValueError
        Inputs or controls are invalid, the taper has zero energy, or no
        usable tiles remain. Zero-valued pixels are allowed.
    """
    int_fwd, int_bwd = _intensities(int_fwd, int_bwd)
    dy, dx = _spacing(pixel_spacing)
    support = np.broadcast_to(np.asarray(window_size, dtype=float), (2,))
    if not np.all(np.isfinite(support)) or np.any(support <= 0):
        raise ValueError('window_size must be positive finite metres')
    if not np.isfinite(tile_outlier_threshold) or tile_outlier_threshold < 0:
        raise ValueError('tile_outlier_threshold must be finite and nonnegative')
    shape = int_fwd.shape
    # A tile covering the full axis cannot grow: avoid searching for an FFT
    # size far larger than the image when the requested support is oversized.
    nx = min(max(2, int(support[1] / dx)), shape[1])
    ny = min(max(2, int(support[0] / dy)), shape[0])
    if nx < shape[1]:
        nx = min(drutils.optimize_fftsize(nx, 3), shape[1])
    if ny < shape[0]:
        ny = min(drutils.optimize_fftsize(ny, 3), shape[0])
    x_starts = np.arange(0, shape[1] - nx + 1, max(1, nx // 2))
    y_starts = np.arange(0, shape[0] - ny + 1, max(1, ny // 2))
    if verbose:
        print(f'Number of windows: {x_starts.size} x {y_starts.size}')
    if win_func is None:
        win_func = np.hamming
    win_x, win_y = np.asarray(win_func(nx)), np.asarray(win_func(ny))
    if (win_x.shape != (nx,) or win_y.shape != (ny,)
            or np.iscomplexobj(win_x) or np.iscomplexobj(win_y)
            or not np.all(np.isfinite(win_x)) or not np.all(np.isfinite(win_y))
            or np.any(win_x < 0) or np.any(win_y < 0)):
        raise ValueError('win_func must return finite nonnegative real 1D weights')
    window = win_y[:, None] * win_x[None, :]
    da = dx * dy
    win_norm = np.sum(window**2 * da**2)
    if win_norm <= 0 or not np.isfinite(win_norm):
        raise ValueError('Window must have positive finite energy')
    if remove_outliers:
        options = dict(thresh=thresh, lp_scale=lp_scale, pixel_spacing=np.sqrt(da))
        int_fwd = outlier_removal(int_fwd, **options)
        int_bwd = outlier_removal(int_bwd, **options)
    fwd_mean, bwd_mean = int_fwd.mean(), int_bwd.mean()
    tiles = np.full((y_starts.size, x_starts.size, ny, nx), np.nan+0j,
                    dtype=np.complex64)
    brightness = np.zeros(tiles.shape[:2])
    valid = np.zeros(tiles.shape[:2], dtype=bool)
    for ix, x_start in enumerate(x_starts):
        for iy, y_start in enumerate(y_starts):
            fwdi = int_fwd[y_start:y_start+ny, x_start:x_start+nx]
            bwdi = int_bwd[y_start:y_start+ny, x_start:x_start+nx]
            fwd_tile_mean, bwd_tile_mean = fwdi.mean(), bwdi.mean()
            brightness[iy, ix] = (fwd_tile_mean + bwd_tile_mean) / 2
            fwd_scale = fwd_tile_mean if tile_normalization else fwd_mean
            bwd_scale = bwd_tile_mean if tile_normalization else bwd_mean
            if fwd_scale <= 0 or bwd_scale <= 0:
                continue
            fwdi = (fwdi - fwd_tile_mean) * window / fwd_scale
            bwdi = (bwdi - bwd_tile_mean) * window / bwd_scale
            spectrum = fft.fft2(fwdi) * da * np.conj(fft.fft2(bwdi) * da)
            tiles[iy, ix] = fft.fftshift(spectrum) / win_norm
            valid[iy, ix] = True
    if not valid.any():
        raise ValueError('No usable cross-spectrum tiles: normalization means are zero')
    # Use pre-centering brightness; comparing each tile with itself would
    # silently accept every tile and measuring centered data loses brightness.
    levels = brightness[valid]
    accepted = valid & (brightness <= levels.mean() + levels.std() * tile_outlier_threshold)
    average = tiles[accepted].mean(axis=0)
    kx = fft.fftshift(fft.fftfreq(nx, dx))
    ky = fft.fftshift(fft.fftfreq(ny, dy))
    return kx, ky, average, tiles


def quick_xspec(int_fwd, int_bwd, *, pixel_spacing, avg=(5, 5),
                remove_outliers=True, lp_scale=1e3, thresh=100.0):
    """Estimate a single-window intensity cross-spectrum for quick inspection.

    Parameters
    ----------
    int_fwd, int_bwd : 2D real array
        Finite nonnegative intensities of equal shape (azimuth, range).
    pixel_spacing : float or pair of float
        Input sampling in metres (azimuth, ground range).
    avg : pair of int, optional
        Spectral smoothing support (azimuth, range), default (5, 5).
        Decimate each resulting axis by ceil(avg/2).
    remove_outliers : bool, optional
        Apply outlier_removal first; default True.
    lp_scale, thresh : float, optional
        Outlier reference support in metres and relative-excess threshold.

    Returns
    -------
    kx, ky : numpy.ndarray
        Decimated range and azimuth wavenumber axes in cycles/m.
    spectrum : numpy.ndarray
        Smoothed complex FFT(fwd) * conj(FFT(bwd)), shape (ky, kx). Each
        image is mean-subtracted, Hann-tapered and zero-padded to powers of
        two. Unlike welch_xspec, no mean-intensity or window-energy
        normalization is applied; units are the product of input units.
    """
    int_fwd, int_bwd = _intensities(int_fwd, int_bwd)
    dy, dx = _spacing(pixel_spacing)
    averaging = np.asarray(avg)
    if (averaging.shape != (2,) or not np.all(np.isfinite(averaging))
            or np.any(averaging < 1) or np.any(averaging != averaging.astype(int))):
        raise ValueError('avg must contain two positive integers (azimuth, range)')
    averaging = averaging.astype(int)
    shape = int_fwd.shape
    padded_shape = tuple(int(drutils.optimize_fftsize(size)) for size in shape)
    window = np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]
    if remove_outliers:
        options = dict(thresh=thresh, lp_scale=lp_scale, pixel_spacing=np.sqrt(dx * dy))
        int_fwd = outlier_removal(int_fwd, **options)
        int_bwd = outlier_removal(int_bwd, **options)
    padded = []
    for image in (int_fwd, int_bwd):
        values = np.zeros(padded_shape, dtype=np.float32)
        values[:shape[0], :shape[1]] = (image - image.mean()) * window
        padded.append(values)
    spectrum = fft.fftshift(fft.fft2(padded[0]) * np.conj(fft.fft2(padded[1])))
    spectrum = drutils.smooth(drutils.smooth(spectrum, int(averaging[0]), axis=0),
                             int(averaging[1]), axis=1)
    strides = np.ceil(averaging / 2).astype(int)
    kx = fft.fftshift(fft.fftfreq(padded_shape[1], dx))[::strides[1]]
    ky = fft.fftshift(fft.fftfreq(padded_shape[0], dy))[::strides[0]]
    return kx, ky, spectrum[::strides[0], ::strides[1]]
