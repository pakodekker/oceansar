# Supporting code for factorizing raw data generation; nothing too fancy here.

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

    return (
        1.5 * params["sr0"] * params["l0"] / params["ant_l_tx"]
        + surface.Ly) / params["v_ground"]


def factorize_raw_params(cfg, params, surface, info, internal_oversampling=8,
                         scansar_guard_coarse_samples=2):
    factorize = cfg.srg.factorize
    prf = cfg.mode.prf
    operation_mode = getattr(cfg.mode, "mode", "stripmap").lower()
    t_span = nominal_raw_time_span(cfg, params, surface)
    if operation_mode == "scansar":
        info.msg("Using ScanSAR burst duration: %f s" % t_span,
                 importance=2)

    if factorize:
        info.msg("Factorizing raw data generation", importance=2)
        # We will compute less surface realizations
        # Coherence time of the surface, for a large area
        tau_c = closure.grid_coherence(
            cfg.ocean.wind_U, 500, params["f0"])
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
                "ScanSAR factorization guard: %d internal pulses; "
                "keeping %d centered pulses"
                % (full_az_steps, requested_az_steps),
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
