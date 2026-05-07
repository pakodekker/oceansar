# Supporting code for factorizing raw data generation; nothing too fancy here.

import numpy as np
import scipy as sp
from oceansar import closure
from oceansar import utils
from tqdm import tqdm


def _next_divisor_at_least(value, minimum):
    """Return the smallest divisor of value that is at least minimum."""
    minimum = max(1, int(np.ceil(minimum)))
    for divisor in range(minimum, value + 1):
        if value % divisor == 0:
            return divisor
    return value


def factorize_raw_params(cfg, params, surface, info, internal_oversampling=8):
    factorize = cfg.srg.factorize
    prf = cfg.sar.prf
    if factorize:
            info.msg("Factorizing raw data generation", importance=2)
            # We will compute less surface realizations
            # Coherence time of the surface, for a large area
            tau_c = closure.grid_coherence(cfg.ocean.wind_U,500, params["f0"])
            info.msg("Surface coherence time: %f s" % (tau_c))
            n_pulses_b = int(utils.optimize_fftsize(int(tau_c * prf/4)))
            info.msg("PRF down-sampling rate =%i" % (n_pulses_b))
            # n_pulses_b = 4
            params["t_step"] = 1./prf
            params["t_span"] = (1.5*(params["sr0"] * params["l0"]/params["ant_l_tx"]) + surface.Ly)/params["v_ground"]  
            

            params["az_steps"] = int(np.floor(params["t_span"]/params["t_step"]))
            az_steps_ = int(np.ceil(params["az_steps"] / n_pulses_b))+1
            #az_steps_ = utils.optimize_fftsize(az_steps_)
            az_steps_ = sp.fft.next_fast_len(az_steps_)
            params["t_span"] = az_steps_ * n_pulses_b * params["t_step"]
            params["t_step"] = params["t_step"] * n_pulses_b
            # Doppler bandwidth for a given block-length
            # length to Doppler bandwidth factor
            ly2dop = 2 * params["v_ground"] / params["l0"] * 1 / params["sr0"]
            # now I make sure this the block is small enough to this to be properly sampled
            # ly2dop * ly < 1/params["t_step"]/internal_oversampling
            block_ly = 1/(params["t_step"]*internal_oversampling*ly2dop)
            info.msg("Block size in azimuth: %f m" % (block_ly))
            # Now adjust the block size to be a divisor of the surface grid
            # length, otherwise block_Ny truncates and the reshape/indexing
            # path below loses the tail rows.
            nblocks = _next_divisor_at_least(surface.Ny, np.ceil(surface.Ly/block_ly))
            block_ly = surface.Ly/nblocks
            info.msg("Adjusted block size in azimuth: %f m" % (block_ly))
            info.msg("Number of blocks: %i" % nblocks)
            params["az_steps"] = az_steps_
            params["block_ly"] = block_ly
            params["nblocks"] = nblocks
            params["block_Ny"] = surface.Ny//nblocks
            params["n_pulses_b"] = n_pulses_b

    else:
        info.msg("Not factorizing raw data generation", importance=2)
        params["t_step"] = 1./prf
        params["t_span"] = (1.5*(params["sr0"] * params["l0"]/params["ant_l_tx"]) + surface.Ly)/params["v_ground"]  
        params["az_steps"] = int(np.floor(params["t_span"]/params["t_step"]))
    params["az0"] = -params["t_span"]*params["v_ground"]/2
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
            rcm_smp = (rcm_b*2/3e8*params["Fs"])[np.newaxis,:, np.newaxis]
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
    if do_hh:
        proc_raw_hh_full = proc_raw_hh_full[:, 0:az_steps_out, :rg_samp]
    if do_vv:        
        proc_raw_vv_full = proc_raw_vv_full[:, 0:az_steps_out, :rg_samp]    
    return proc_raw_hh_full, proc_raw_vv_full
