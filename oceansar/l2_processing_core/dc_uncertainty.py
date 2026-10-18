"""Spectral Cramer--Rao prediction for proper complex Gaussian speckle.

This is a large-observation (Whittle) approximation, not a finite-covariance
bound. Supply the nominal acquired spectrum and its physical-parameter
derivative, before estimator equalization. Fixed support edges do not move.
"""

import numpy as np


def doppler_crlb(frequency, power, derivative, *, observation_time_s,
                 independent_range_looks=1., noise_density=0., unknown_power=True):
    """Predict the spectral Doppler standard-deviation bound in Hz.

    Parameters
    ----------
    frequency : array-like
        Strictly increasing frequencies in Hz on the final axis (at least
        three points). Leading dimensions identify separate local models.
        Include only the fixed acquired frequency interval, without padded
        zeros or derivatives of its edges.
    power, derivative : array-like
        Nominal signal power density and its signed derivative with respect
        to physical Doppler shift, in power-density/Hz. Broadcast with
        frequency. For A(f-d), supply -A'(f), not the derivative of a fixed
        acquisition gate or processing filter.
    observation_time_s : float or array-like
        Observed azimuth duration in seconds, broadcast to leading model
        dimensions. This is physical duration, not pulse/sample count.
    independent_range_looks : float or array-like, optional
        Independent range cells, default 1. Account for range bandwidth,
        rather than counting oversampled range pixels.
    noise_density : float or array-like, optional
        Known receiver-noise PSD in the same units as power, default 0
        (speckle limit). It does not shift with Doppler. Transform signal,
        derivative AND noise together for an invariance check.
    unknown_power : bool, optional
        Default True: eliminate an unknown common log signal-power scale
        using the two-parameter Fisher matrix. False assumes known power.

    Returns
    -------
    dict of numpy.ndarray
        std_hz, variance_hz2 and efficient information in Hz^-2; unprojected
        fisher_dd, fisher_dlog_power and fisher_log_power_log_power; acquired
        bandwidth_hz, azimuth_modes=T*B and independent_modes=T*B*range_looks.
        Shapes are the leading model dimensions. Unidentifiable shift gives
        infinite uncertainty, never zero uncertainty. To combine independent
        models with a COMMON unknown power, sum their unprojected matrices
        before eliminating that nuisance parameter.

    Notes
    -----
    Proper complex spectral coefficients give I_dd=T*L*integral((P_d/S)^2 df),
    S=P+N, without the one-half factor used for real Gaussian data. Fourier
    cells have spacing 1/T: oversampling a fixed spectrum does not create
    more cells. No PRF or numerical quadrature-point count multiplies I.
    This approximation requires many azimuth modes and a regular positive
    spectrum. Spectral zeros require a finite-covariance/noise model; no
    artificial PSD floor is introduced here. It predicts an unbiased optimal
    estimator's floor, not the actual CDE/SDE error, bias or ensemble SEM.
    """
    if not isinstance(unknown_power, (bool, np.bool_)):
        raise ValueError('unknown_power must be boolean')
    f, p, dp, noise = np.broadcast_arrays(
        np.asarray(frequency, float), np.asarray(power, float),
        np.asarray(derivative, float), np.asarray(noise_density, float))
    if (f.ndim < 1 or f.shape[-1] < 3 or not np.all(np.isfinite(f))
            or np.any(np.diff(f, axis=-1) <= 0) or not np.all(np.isfinite(p))
            or np.any(p < 0) or not np.all(np.isfinite(dp))
            or not np.all(np.isfinite(noise)) or np.any(noise < 0)
            or np.any(p+noise <= 0)):
        raise ValueError('Need increasing finite frequencies, nonnegative PSDs, positive total PSD and finite derivatives')
    duration, looks = np.broadcast_arrays(np.asarray(observation_time_s, float),
                                          np.asarray(independent_range_looks, float),
                                          np.empty(f.shape[:-1]))[:2]
    if (not np.all(np.isfinite(duration)) or np.any(duration <= 0)
            or not np.all(np.isfinite(looks)) or np.any(looks <= 0)):
        raise ValueError('Observation time and independent range looks must be positive finite')
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    score, amplitude_score = dp/(p+noise), p/(p+noise)
    scale = duration*looks
    dd = scale*integrate(score**2, f, axis=-1)
    da = scale*integrate(score*amplitude_score, f, axis=-1)
    aa = scale*integrate(amplitude_score**2, f, axis=-1)
    if unknown_power:
        coefficient = np.divide(da, aa, out=np.zeros_like(da), where=aa > 0)
        # Integrate the residual score instead of subtracting almost equal
        # Fisher entries: a pure brightness change carries no shift information.
        info = scale*integrate((score-coefficient[..., None]*amplitude_score)**2,
                                f, axis=-1)
        info = np.where(info > 64*np.finfo(float).eps*dd, info, 0.)
    else:
        info = dd
    variance = np.divide(1., info, out=np.full_like(info, np.inf), where=info > 0)
    bandwidth = f[..., -1]-f[..., 0]
    return dict(std_hz=np.sqrt(variance), variance_hz2=variance, information=info,
                fisher_dd=dd, fisher_dlog_power=da, fisher_log_power_log_power=aa,
                bandwidth_hz=bandwidth, azimuth_modes=duration*bandwidth,
                independent_modes=duration*looks*bandwidth)
