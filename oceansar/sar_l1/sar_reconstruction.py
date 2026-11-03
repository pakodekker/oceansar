import json
import numpy as np
from oceansar import ocs_io as tpio
from scipy.signal import resample_poly
from oceansar import constants as const
from oceansar.ocs_io import product_metadata as metadata
from oceansar.utils import geometry as geo
#create a function
def raw_reconstr(raw_output_file, reconstr_output_file, K, channel_ids = [0, 1, 2, 3, 4], downsample = True):
    # Load config and raw data
    """Reconstruct an azimuth-oversampled raw signal from along-track channels.

    Parameters
    ----------
    raw_output_file : str or path-like
        RawFile with complex samples ordered (polarization, channel, azimuth,
        range), along-track baselines in metres and radar/geometry metadata.
        Only the first polarization is used. The current frequency-domain
        matrix inversion requires an odd number of channels and invertible
        channel transfer matrices.
    reconstr_output_file : str or path-like
        Destination RawFile with complex sample shape
        (1, 1, number_of_channels * input_azimuth_samples, range_samples).
        The original acquisition PRF and reconstructed azimuth sampling rate
        are both recorded in the product metadata.
    K: int
        Number of ambiguities to be reconstructed. The output azimuth sample count is
        ``K * input_azimuth_samples``. The output azimuth sampling rate is
        ``K * input_PRF``. The output azimuth frequency grid is
        ``[-K/2 * input_PRF, K/2 * input_PRF]``.
    channel_ids : sequence of int, optional
        Input channels used for reconstruction. The selected IDs are saved as
        the ``ids_ch`` product variable.
    downsample : bool, optional
        If True, the reconstructed signal is downsampled to the original azimuth sample count. 
        The output azimuth sampling rate is the same as the input PRF.

    Returns
    -------
    None
        Write reconstructed samples and copied geometry metadata to disk.

    Notes
    -----
    Apply the desired system errors before reconstruction. This function
    neither calibrates samples nor adds receiver noise.
    """
    # REVIEW (Yan Yuan): exceptions before close() leave datasets open. Suggested
    # structure: wrap each dataset's processing in try/finally, including output.
    # raw_file = tpio.RawFile(raw_output_file, 'r')
    # try:
    #     ...  # Read and reconstruct while the input dataset is open.
    # finally:
    #     raw_file.close()
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')  # Shape: [num_bursts, num_ch, az_size, rg_size]
    sr0 = raw_file.get('sr0')
    az0 = raw_file.get('az0')
    inc_angle = raw_file.get('inc_angle')
    b_ati = raw_file.get('b_ati')
    ant_L = raw_file.get('ant_L')

    print(f"Raw data shape: {raw_data.shape}")

    # Extract parameters
    # Load the processed data
    f0 = raw_file.get('f0')
    # REVIEW (Yan Yuan): distinguish acquisition PRF from actual input sampling.
    # Resampled products can have az_sampling != prf. Use sample_rate throughout
    # FFT/alias grids, df and output time grids; retain acquisition_prf in headers.
    # acquisition_prf = metadata.scalar(raw_file.__file__, 'prf', positive=True)
    # sample_rate = metadata.azimuth_sampling(raw_file.__file__)
    prf = raw_file.get('prf')
    rg_bw = raw_file.get('rg_bw')
    rg_sampling = raw_file.get('rg_sampling')
    v_ground = raw_file.get('v_ground')
    alt = raw_file.get('orbit_alt')

    # REVIEW (Yan Yuan): this altitude-based speed may differ from the GeoHistory
    # velocity used to generate channel phases. Suggested fix: restore geometry
    # from input metadata and derive the same orbital velocity/angle mapping as
    # the generator; validate any constant-speed approximation over the aperture.
    # geometry_cfg = metadata.focusing_configuration(cfg, raw_file.__file__)
    # Requires a cfg argument (or a geometry factory); cfg is not available here.
    v_orbit = geo.orbit_to_vel(alt, inc=np.deg2rad(inc_angle))

    l0 = const.c / f0

    print(f"v_orbit: {v_orbit}, prf: {prf}")

    # let's start from following the paper
    # REVIEW (Yan Yuan): nonzero XTI introduces phase not modeled in H_vec.
    # Suggested interim fix: reject selected XTI channels until their phase is
    # compensated with a range-dependent model, rather than silently ignoring it.
    # b_xti = np.asarray(raw_file.get('b_xti'))[channel_ids]
    # if np.any(b_xti != 0):
    #     raise NotImplementedError('Reconstruction requires zero XTI baselines')
    b_ati = b_ati[channel_ids]
    raw_data = raw_data[:, channel_ids, :, :]
    N_ch = raw_data.shape[1] # number of channels
    N_az = raw_data.shape[2]
    N_rg = raw_data.shape[3]
    assert 1 <= K <= N_ch, "K must be between 1 and N_ch"
    # # construct transfer function in frequency domain, eq13 (Krieger et al, 2004)
    f_dop = np.fft.fftshift(np.fft.fftfreq(raw_data.shape[2], d=1./prf))
    k_start = np.ceil(-K / 2 - f_dop / prf).astype(int)
    k_alias = k_start[:, None] + np.arange(K)[None, :]
    f_matrix = f_dop[:, None] + k_alias * prf
    # REVIEW (Yan Yuan): derive this transfer function against the simulator's
    # phase-only exp(-1j*k0*b_ati*sin_az) channel model. The b_ati**2 term is not
    # explicit there; its validity/sign needs a stationary-phase derivation and
    # an end-to-end synthetic test before changing it. Candidate ONLY if that
    # derivation confirms no quadratic baseline phase for this generator:
    # H_vec = np.exp(-1j*np.pi*b_ati[None, :, None]
    #                * f_matrix[:, None, :] / v_orbit)
    H_vec = np.exp(
        -1j * np.pi * (
            b_ati[None, :, None]**2 / (2 * l0 * sr0)
            + b_ati[None, :, None] * f_matrix[:, None, :] / v_orbit
        )
    ) 
    # H_vec = np.exp(-1j * (v_ground/v_orbit) *np.pi * (b_ati**2 / (2 * l0 * sr0) + b_ati * f_matrix[:,:, None] / v_orbit)) 
    # REVIEW (Yan Yuan): inversion may amplify noise severely for unfavorable
    # channel spacing; pinv can silently truncate unrecoverable ambiguity modes.
    # Suggested guard (threshold is a policy choice, not a validated constant):
    # singular_values = np.linalg.svd(H_vec, compute_uv=False)
    # condition = singular_values[:, 0] / singular_values[:, -1]
    # condition_limit = ...  # Choose from acceptable noise amplification.
    # if np.any(~np.isfinite(condition)) or np.any(condition > condition_limit):
    #     raise ValueError('Channel reconstruction matrix is ill-conditioned')
    # Consider noise-aware regularization instead of rejection if appropriate.
    if K == N_ch:
        P_vec = K * np.linalg.inv(H_vec)
    else:
        P_vec = K * np.linalg.pinv(H_vec)
    raw_data_fft = np.fft.fftshift(np.fft.fft(raw_data[0, :, :, :], axis=1), axes = 1) # FFT along azimuth
    reconstr_signal = np.einsum('car,abc->bar', raw_data_fft, P_vec)
    N_out = K * N_az
    df = prf / N_az

    f_out = np.fft.fftshift(
        np.fft.fftfreq(N_out, d=1.0 / (K * prf))
    )

    # Map each recovered component to its actual output frequency bin
    indices = np.rint(
        (f_matrix - f_out[0]) / df
    ).astype(int)

    assert np.all((indices >= 0) & (indices < N_out)), (
        "Some reconstructed frequencies lie outside the output grid."
    )

    assert np.unique(indices).size == N_out, (
        "Frequency mapping is not one-to-one; check alias selection."
    )

    full_spectrum = np.zeros((N_out, N_rg), dtype=complex)

    for az_idx in range(N_az):
        for band_idx in range(K):
            out_idx = indices[az_idx, band_idx]
            full_spectrum[out_idx, :] = reconstr_signal[band_idx, az_idx, :]

    # Convert to the azimuth domain
    upsample_signal = np.fft.ifft(np.fft.ifftshift(full_spectrum, axes=0), axis=0)
    # add the dimension of polarization, number of channels
    if downsample:
        downsample_signal = resample_poly(
            upsample_signal,
            up=1,
            down=K,
            axis=0
            )
        reconstr_data = downsample_signal[None, None, :, :]
    else:
        reconstr_data = upsample_signal[None, None, :, :]
        source_az = np.arange(N_az)
        output_az = np.arange(N_out) / K
    print('Reconstruction completed!')
    # These vectors follow the azimuth sample grid, so interpolate them to
    # match the reconstructed signal's increased sample count.
    sr_pt = np.ma.asarray(raw_file.get('sr_pt'))
    if (sr_pt.shape != (N_az,) or np.ma.is_masked(sr_pt)
            or not np.isfinite(sr_pt).all()):
        raise ValueError('Raw sr_pt must contain one finite value per azimuth sample')

    nrcs_avg = np.ma.asarray(raw_file.get('NRCS_avg'))
    if (nrcs_avg.shape != (raw_data.shape[0], N_az)
            or np.ma.is_masked(nrcs_avg)
            or not np.isfinite(nrcs_avg).all()):
        raise ValueError('Raw NRCS_avg must match the polarization and azimuth dimensions')

    source_labels = json.loads(
        getattr(raw_file.__file__, 'polarizations', '[]'))
    if source_labels:
        if len(source_labels) != raw_data.shape[0]:
            raise ValueError('Raw polarization labels do not match the polarization dimension')
        output_labels = [source_labels[0]]
    else:
        output_labels = ['pol0']

    # Preserve annotations that sar_focus uses to restore acquisition context.
    copied_attributes = {
        name: raw_file.__file__.getncattr(name)
        for name in (
            'antenna_configuration', 'antenna_pattern_scope',
            'processing_geometry', 'acquisition_mode',
            'burst_image_length_m', 'point_targets_follow_current',
            'point_target_current_vector_m_s', 'point_target_elapsed_start_s',
            'factorization_azimuth_padding_full_prf_samples')
        if name in raw_file.__file__.ncattrs()
    }
    raw_file.close()

    # Write the standard four-dimensional raw schema expected by sar_focus.
    # REVIEW (Yan Yuan): mode='w' overwrites an existing destination. Suggested
    # preflight before opening input/output (requires `from pathlib import Path`):
    # if Path(reconstr_output_file).exists():
    #     raise FileExistsError(reconstr_output_file)
    # Also use try/finally for this dataset, as suggested for the input above.
    reconstr_file = tpio.RawFile(reconstr_output_file, 'w', reconstr_data.shape)
    reconstr_file.set('inc_angle', inc_angle)
    reconstr_file.set('f0', f0)
    reconstr_file.set('num_ch', reconstr_data.shape[1])
    reconstr_file.set('ant_L', ant_L)
    # REVIEW (Yan Yuan): after separating the two rates above, output headers:
    # output_rate = sample_rate if downsample else K*sample_rate
    # metadata.write_sampling(reconstr_file.__file__, output_rate, acquisition_prf)
    if downsample:
        metadata.write_sampling(reconstr_file.__file__, prf, prf)
        reconstr_file.set('sr_pt', sr_pt)
        # REVIEW (Yan Yuan): output has one polarization, but dual-pol nrcs_avg
        # has two rows and raises a broadcasting error here. Suggested replacement:
        # reconstr_file.set('NRCS_avg', nrcs_avg[:1, :])
        reconstr_file.set('NRCS_avg', nrcs_avg)
    else:
        metadata.write_sampling(reconstr_file.__file__, K * prf, prf)
        # REVIEW (Yan Yuan): output_az extends beyond source_az[-1] by (K-1)/K.
        # np.interp holds sr_pt constant for these last samples. Prefer evaluating
        # the saved range model on output times. A fallback linear extrapolation
        # (requires N_az >= 2 and validation of endpoint accuracy) would be:
        # sr_pt_out = np.interp(output_az, source_az, sr_pt)
        # tail = output_az > source_az[-1]
        # sr_pt_out[tail] = (sr_pt[-1] + (output_az[tail]-source_az[-1])
        #                    * (sr_pt[-1]-sr_pt[-2]))
        # reconstr_file.set('sr_pt', sr_pt_out)
        reconstr_file.set('sr_pt', np.interp(output_az, source_az, sr_pt))
        reconstr_file.set('NRCS_avg', np.interp(output_az, source_az, nrcs_avg[0])[None, :])
    reconstr_file.set('v_ground', v_ground)
    reconstr_file.set('az0', az0)
    reconstr_file.set('orbit_alt', alt)
    reconstr_file.set('sr0', sr0)
    reconstr_file.set('rg_sampling', rg_sampling)
    reconstr_file.set('rg_bw', rg_bw)
    reconstr_file.set('b_ati', np.zeros(1))
    reconstr_file.set('b_xti', np.zeros(1))
    reconstr_file.set('raw_data*', reconstr_data)

    ids_ch = np.asarray(channel_ids, dtype=np.int32)
    reconstr_file.__file__.createDimension('ids_ch_dim', ids_ch.size)
    ids_ch_var = reconstr_file.__file__.createVariable(
        'ids_ch', 'i4', ('ids_ch_dim',))
    ids_ch_var.units = '[]'
    ids_ch_var[:] = ids_ch
    reconstr_file.__file__.polarizations = json.dumps(output_labels)
    for name, value in copied_attributes.items():
        reconstr_file.__file__.setncattr(name, value)
    reconstr_file.close()
