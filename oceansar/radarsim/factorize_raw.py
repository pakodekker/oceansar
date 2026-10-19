"""Block range references and interpolation for factorized SAR raw generation.

Remove common satellite motion and deterministic current before constructing
coarse range profiles; restore the same phase and range delay at full PRF.
"""

import numpy as np
import scipy as sp
from oceansar import closure
from oceansar import constants as const
from tqdm import tqdm


def _next_divisor_at_least(value, minimum):
    """Return the smallest divisor of value that is at least minimum."""
    minimum = max(1, int(np.ceil(minimum)))
    for divisor in range(minimum, value + 1):
        if value % divisor == 0:
            return divisor
    return value


def nominal_raw_time_span(cfg, params, surface):
    """Return the requested acquisition duration before discretization."""
    operation_mode = getattr(cfg.mode, "mode", "stripmap").lower()
    if operation_mode == "scansar":
        t_span = getattr(cfg.mode, "t_burst", None)
        if t_span is None or t_span <= 0:
            raise ValueError("ScanSAR mode requires a positive t_burst")
        return t_span

    aperture_time = getattr(cfg.mode, 'aperture_time', None)
    if aperture_time is not None:
        if not np.isfinite(aperture_time) or aperture_time <= 0:
            raise ValueError('mode.aperture_time must be a positive duration in seconds')
        return float(aperture_time)
    sine_span = params.get('antenna_sine_span', 1.5*params['l0']/params['ant_l_tx'])
    return (params['sr0']*sine_span + surface.Ly) / params['v_ground']


def factorization_scene_time(cfg, params, elapsed_time):
    """Return ocean time in seconds for elapsed generator time in seconds.

    cfg.ocean.frozen_ocean selects a fixed params['frozen_scene_time']
    snapshot (default zero). Otherwise add params['scene_time_offset']
    (default zero) to elapsed_time. Scalars and arrays are accepted. Use this
    same clock for materializing the ocean and constructing both reference
    grids: guards may extend numerical time without shifting the scene epoch.
    """
    if cfg.ocean.frozen_ocean:
        return np.zeros_like(elapsed_time, dtype=float)+params.get('frozen_scene_time', 0.)
    return np.asarray(elapsed_time, dtype=float)+params.get('scene_time_offset', 0.)


def factorization_current_range(cfg, params, surface, elapsed_time,
                                 sin_incidence, sin_azimuth):
    """Return deterministic current displacement projected into slant range.

    Parameters
    ----------
    cfg : ConfigFile
        ocean.frozen_ocean controls the scene clock; optional
        srg.factorize_current (default True) enables this reference term.
    params : dict
        Scene-clock offsets used by factorization_scene_time, in seconds.
    surface : OceanSurface
        Supplies current=(Ux, Uy) in m/s in the surface's range/azimuth axes.
        Use the loaded ocean's current, rather than possibly changed config.
    elapsed_time : float or array
        Generator time from the first INTERNAL sample, in seconds, evaluated
        separately on the coarse and full-PRF grids.
    sin_incidence, sin_azimuth : float or array
        Projection factors of the common reference point. Incidence uses the
        first range column, matching the platform range reference; azimuth
        uses the same GeoHistory direction and block centre as the actual
        surface phase. Arrays must broadcast with the scene time.

    Returns
    -------
    float or numpy.ndarray
        Range offset t_scene*(Ux*sin_incidence + Uy*sin_azimuth), in metres.
        Disabled correction returns zero with the same broadcast shape.

    Notes
    -----
    OceanSurface already includes U*t_scene in Dx/Dy, and the raw phase is
    exp(-2j*k0*range). ADD this term to the removed block range reference
    before generating coarse profiles, then restore the SAME reference on
    the full grid. This demodulates the known carrier AND removes its range
    drift before interpolation, without changing physical current or the
    random sequence, coherence time, coarse sampling rate or guard count.
    At broadside, f_current=-2*Ux*sin(theta)/lambda. A frozen snapshot has no
    growing U*t term; its constant displacement is still projected through
    the time-varying azimuth geometry. Residual within-block/swath projection
    differences remain in the coarse fields.
    """
    ux, uy = np.asarray(surface.current, dtype=float)
    time = factorization_scene_time(cfg, params, elapsed_time)
    result = time*(ux*np.asarray(sin_incidence)+uy*np.asarray(sin_azimuth))
    return result if getattr(cfg.srg, 'factorize_current', True) else np.zeros_like(result)


def factorize_raw_params(cfg, params, surface, info, internal_oversampling=8,
                         scansar_guard_coarse_samples=None):
    """Derive coarse sampling, spatial blocks and retained raw pulse support.

    cfg supplies acquisition mode/PRF, ocean wind speed and srg.factorize.
    params supplies carrier/wavelength, reference range, antenna length and
    ground speed; surface supplies azimuth extent and grid count. Return the
    updated params dict with internal/coarse time step in seconds, integer
    decimation, block dimensions and output crop/origin. Coarse sampling is
    determined by stochastic ocean coherence; by default deterministic current
    is removed by factorization_current_range and restored during aggregation.

    scansar_guard_coarse_samples is an optional nonnegative integer override.
    None reads cfg.srg.scansar_guard_coarse_samples, defaulting to 8. This is
    the requested extra COARSE samples per side for factorized ScanSAR only:
    add twice this count before rounding the coarse grid to a fast FFT length,
    then crop the interpolated data back to the requested burst. Increasing
    guards extends internal support without changing the coarse sampling rate
    or retained pulse count. Explicit overrides serve controlled experiments.
    params['scansar_guard_coarse_samples'] records the resolved requested count;
    az_steps, n_pulses_b and az_crop_start describe the actual padded grid.
    """
    factorize = cfg.srg.factorize
    prf = cfg.mode.prf
    operation_mode = getattr(cfg.mode, "mode", "stripmap").lower()
    t_span = nominal_raw_time_span(cfg, params, surface)
    if operation_mode == "scansar":
        info.msg("Using ScanSAR burst duration: %f s" % t_span,
                 importance=2)

    if factorize:
        if operation_mode == 'scansar':
            if scansar_guard_coarse_samples is None:
                scansar_guard_coarse_samples = getattr(
                    cfg.srg, 'scansar_guard_coarse_samples', 8)
            guards = scansar_guard_coarse_samples
            if (isinstance(guards, (bool, np.bool_)) or not np.isscalar(guards)
                    or not isinstance(guards, (int, float, np.integer, np.floating))
                    or not np.isfinite(guards) or guards < 0 or int(guards) != guards):
                raise ValueError('scansar_guard_coarse_samples must be a nonnegative integer')
            scansar_guard_coarse_samples = int(guards)
            params['scansar_guard_coarse_samples'] = scansar_guard_coarse_samples
        info.msg("Factorizing raw data generation", importance=2)
        # We will compute less surface realizations
        # Coherence time of the surface, for a large area
        tau_c = closure.grid_coherence(
            cfg.ocean.wind_U, 500, params["f0"])
        # This sets sampling for the stochastic residual, not its deterministic
        # current carrier. The block reference removes that known carrier; do
        # not alter the physical coherence model to mimic frequency translation.
        info.msg("Surface coherence time: %f s" % tau_c)
        n_pulses_b = sp.fft.next_fast_len(int(tau_c * prf/4))
        info.msg("PRF down-sampling rate =%i" % n_pulses_b)
        params["t_step"] = 1./prf
        params["t_span"] = t_span
        params["az_steps"] = int(np.floor(t_span/params["t_step"]))
        requested_az_steps = params["az_steps"]
        if operation_mode == "scansar":
            az_steps = (
                int(np.ceil(requested_az_steps / n_pulses_b))
                + 2 * scansar_guard_coarse_samples
            )
        else:
            az_steps = int(np.ceil(requested_az_steps / n_pulses_b)) + 1
        az_steps = sp.fft.next_fast_len(az_steps)
        full_az_steps = az_steps * n_pulses_b
        params["t_span"] = full_az_steps * params["t_step"]
        params["t_step"] *= n_pulses_b
        if operation_mode == "scansar":
            params["output_az_steps"] = requested_az_steps
            params["az_crop_start"] = (
                full_az_steps - requested_az_steps + 1) // 2
            info.msg(
                "ScanSAR factorization: %d requested coarse guards per side, "
                "%d coarse samples after FFT padding; "
                "%d internal pulses, keeping %d centered pulses"
                % (scansar_guard_coarse_samples, az_steps,
                   full_az_steps, requested_az_steps),
                importance=2)
        else:
            params["output_az_steps"] = full_az_steps
            params["az_crop_start"] = 0
        # Doppler bandwidth for a given block-length
        ly2dop = (2 * params["v_ground"] / params["l0"]
                  / params["sr0"])
        block_ly = 1/(params["t_step"]*internal_oversampling*ly2dop)
        info.msg("Block size in azimuth: %f m" % block_ly)
        # Make the block size divide the surface grid exactly.
        nblocks = _next_divisor_at_least(
            surface.Ny, np.ceil(surface.Ly/block_ly))
        block_ly = surface.Ly/nblocks
        info.msg("Adjusted block size in azimuth: %f m" % block_ly)
        info.msg("Number of blocks: %i" % nblocks)
        params["az_steps"] = az_steps
        params["block_ly"] = block_ly
        params["nblocks"] = nblocks
        params["block_Ny"] = surface.Ny//nblocks
        params["n_pulses_b"] = n_pulses_b
    else:
        info.msg("Not factorizing raw data generation", importance=2)
        params["t_step"] = 1./prf
        params["t_span"] = t_span
        params["az_steps"] = int(np.floor(t_span/params["t_step"]))
        params["output_az_steps"] = params["az_steps"]
        params["az_crop_start"] = 0
    params["az0"] = -params["t_span"]*params["v_ground"]/2
    params["output_az0"] = (
        params["az0"]
        + params["az_crop_start"] / prf * params["v_ground"]
    )
    return params


def aggregate_factorized_raw(proc_raw_hh, proc_raw_vv, 
                            sr_surface_fct, sr_surface_fct_full,
                            params, surface, cfg, info, workers=4):
    """Interpolate block residual fields and restore their full range reference.

    Parameters
    ----------
    proc_raw_hh, proc_raw_vv : complex ndarray or None
        Coarse residual range profiles, axes (channel, coarse time, block,
        range). None disables that polarization.
    sr_surface_fct, sr_surface_fct_full : ndarray
        Removed reference ranges in metres on coarse/full time grids, axes
        (time, block). Both contain the same platform/current convention.
        The coarse reference is retained in the interface for auditing; the
        already-demodulated input fields need no further coarse correction.
    params : dict
        Block count, coarse pulse count, integer interpolation factor, k0 in
        rad/m, range sampling Fs in Hz and retained full-rate pulse crop.
    surface, cfg : object
        Generator context retained by the strategy interface. Restoration
        uses the explicitly supplied reference arrays, including current.
    info : PrInfo
        Progress logger. workers specifies the SciPy FFT worker count.
    workers : int, optional
        FFT worker count, default 4.

    Returns
    -------
    tuple of complex64 ndarray or None
        HH/VV raw fields with axes (channel, retained full-rate time, range).
        Restore exp(-2j*k0*reference) AND the fractional range delay after
        interpolation, sum blocks and crop to the requested acquisition.
    """
    # Now we need to upsample, and restore the full RCM and phase
    # We will do this block by block, to save memory
    info.msg("Interpolate and restore full phase and RCM")
    nblocks = params["nblocks"]
    az_steps = params["az_steps"]
    if proc_raw_vv is None:
        rg_samp = proc_raw_hh.shape[-1]
    else:
        rg_samp = proc_raw_vv.shape[-1]
    rg_samp_zp = sp.fft.next_fast_len(rg_samp)
    # Output pulses
    az_steps_out = params["az_steps"] * params["n_pulses_b"]
    #az_steps_zp = utils.optimize_fftsize(az_steps_out)
    do_hh = proc_raw_hh is not None
    do_vv = proc_raw_vv is not None
    proc_raw_hh_full = None
    proc_raw_vv_full = None
    if do_hh:
        proc_raw_hh_full = np.zeros([proc_raw_hh.shape[0], az_steps_out, rg_samp_zp], dtype=np.complex64)
        proc_raw_hh_block = np.zeros([proc_raw_hh.shape[0], az_steps_out, rg_samp_zp], dtype=np.complex64)

    if do_vv:
        proc_raw_vv_full = np.zeros([proc_raw_vv.shape[0], az_steps_out, rg_samp_zp], dtype=np.complex64)
        proc_raw_vv_block = np.zeros([proc_raw_vv.shape[0], az_steps_out, rg_samp_zp], dtype=np.complex64)
    #rg_freq = (np.fft.fftfreq(rg_samp_zp))[np.newaxis, np.newaxis, :]
    rg_freq = np.fft.fftfreq(rg_samp_zp).astype(np.float32)[np.newaxis, np.newaxis, :]

    for b in tqdm(range(nblocks)):
            # We need to restore the full RCM and phase for each block, and then aggregate
            # First we need to upsample the raw data for this block
            proc_raw_hh_b = None
            proc_raw_vv_b = None
            # Now something not super nice, I will interpolate sr_surface_fct to the full azimuth grid, and then apply the phase correction 
            rcm_b = sr_surface_fct_full[:,b]  
            # Preserve float64 range/phase until AFTER the complex exponential.
            # Casting a satellite-scale range to float32 here loses sub-wavelength
            # information; storing the resulting unit phasor in complex64 is safe.
            phase_b = - 2 * params["k0"] * rcm_b
            #phasor_b = np.exp(1j*phase_b)
            phasor_b = np.exp(1j * phase_b).astype(np.complex64)
            rcm_smp = (rcm_b * 2 / const.c * params["Fs"])[
                np.newaxis, :, np.newaxis]
            range_phasor_b = np.exp(-1j * 2 * np.pi * rcm_smp * rg_freq).astype(np.complex64)

            if do_hh:
                # We are going to upsample this by zero-padding in the Fourier domain, which is equivalent to sinc interpolation in the time domain
                # So, take block, fft in azimuth, zero-pad, ifft
                proc_raw_hh_b = proc_raw_hh[:, :, b, :]
                proc_raw_hh_block[:] = 0
                proc_raw_hh_block[:,0:az_steps,0:rg_samp_zp] = params["n_pulses_b"] * sp.fft.fftshift(sp.fft.fft(proc_raw_hh_b, axis=1, workers=workers), axes=(1,))
                proc_raw_hh_block = sp.fft.ifft(np.roll(proc_raw_hh_block, shift=-int(az_steps/2), axis=1), axis=1, workers=workers)
                # Now we need to restore the RCM and phase for this block, which is equivalent to multiplying by a complex exponential in the time domain
                # The RCM is given by sr_surface_fct[b], and the phase is given
                proc_raw_hh_block *= phasor_b[np.newaxis,:,np.newaxis]
                proc_raw_hh_block = sp.fft.fft(proc_raw_hh_block, axis=2, workers=workers)
                proc_raw_hh_block *=  range_phasor_b
                proc_raw_hh_block = sp.fft.ifft(proc_raw_hh_block, axis=2, workers=workers)
                proc_raw_hh_full +=  proc_raw_hh_block
                
            if do_vv:
                proc_raw_vv_b = proc_raw_vv[:, :, b, :]
                proc_raw_vv_block[:] = 0
                proc_raw_vv_block[:,0:az_steps,0:rg_samp_zp] = params["n_pulses_b"] * sp.fft.fftshift(sp.fft.fft(proc_raw_vv_b, axis=1, workers=workers), axes=(1,))
                proc_raw_vv_block = sp.fft.ifft(np.roll(proc_raw_vv_block, shift=-int(az_steps/2), axis=1), axis=1, workers=workers)
                proc_raw_vv_block *= phasor_b[np.newaxis,:,np.newaxis]
                proc_raw_vv_block = sp.fft.fft(proc_raw_vv_block, axis=2, workers=workers)
                proc_raw_vv_block *=  range_phasor_b
                proc_raw_vv_block = sp.fft.ifft(proc_raw_vv_block, axis=2, workers=workers)
                proc_raw_vv_full +=  proc_raw_vv_block
    crop_start = params["az_crop_start"]
    crop_stop = crop_start + params["output_az_steps"]
    if do_hh:
        proc_raw_hh_full = proc_raw_hh_full[
            :, crop_start:crop_stop, :rg_samp]
    if do_vv:
        proc_raw_vv_full = proc_raw_vv_full[
            :, crop_start:crop_stop, :rg_samp]
    return proc_raw_hh_full, proc_raw_vv_full
