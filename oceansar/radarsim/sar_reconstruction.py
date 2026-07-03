import numpy as np
from oceansar import ocs_io as tpio
from oceansar import constants as const
from oceansar.utils import geometry as geo
#create a function
def raw_reconstr(raw_output_file, reconstr_output_file):
    # Load config and raw data
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
    prf = raw_file.get('prf')
    num_ch = raw_file.get('num_ch')
    rg_bw = raw_file.get('rg_bw')
    rg_sampling = raw_file.get('rg_sampling')
    v_ground = raw_file.get('v_ground')
    alt = raw_file.get('orbit_alt')

    if v_ground == 'auto':
        v_ground = geo.orbit_to_vel(alt, ground=True)

    l0 = const.c / f0

    print(f"v_ground: {v_ground}, prf: {prf}")

    # let's start from following the paper
    N_ch = raw_data.shape[1] # number of channels
    # construct transfer function in frequency domain, eq13
    H_vec = np.zeros((N_ch, N_ch), dtype=complex)
    for ii in np.arange(N_ch):
        f = f0 + ii * prf
        H_vec[ii, :] = np.exp(-1j * np.pi * (b_ati**2 / (2 * l0 * sr0) + b_ati * f / v_ground)) 
    P_vec = np.linalg.inv(H_vec)

    raw_data_fft = np.fft.fft(raw_data[0, :, :, :], axis=1)  # FFT along azimuth
    raw_data_fft = np.fft.fftshift(raw_data_fft, axes=1)  # Shift zero frequency to center
    freqs = np.fft.fftfreq(raw_data.shape[2], d=1./prf)
    freqs = np.fft.fftshift(freqs)  # Shift frequencies to match the FFT shift
    reconstr_signal = np.zeros_like(raw_data[0, :, :, :], dtype=complex)  # Single channel output
    freqs_bin = prf
    for ii in np.arange(N_ch):
        for jj in np.arange(N_ch): # this is actually number of different frequency bands
            freqs_array = np.where((freqs >= -num_ch * prf/2 + jj * freqs_bin) & (freqs < -num_ch * prf/2 + (jj + 1)* freqs_bin), 1, 0)
            reconstr_signal[ii, :, :] = reconstr_signal[ii, :, :] + raw_data_fft[ii, :, :] * freqs_array[:, None] * P_vec[ii, jj]
    # reconstr_signal = N_ch * np.fft.ifft(np.fft.ifftshift(reconstr_signal, axes=1), axis=1)  # IFFT to get back to time domain
    # combination of muti-channel signals by simply suming up 
    # reconstr_signal_sum = np.sum(reconstr_signal, axis=0)[None, :]
    # apply other methods to combine all channels, e.g., weighted sum 
    print('Reconstruction completed!')  
    # upsampling: placing the recovered ambiguity bands into their correct Doppler locations
    Naz = reconstr_signal.shape[1]
    Nrg = reconstr_signal.shape[2]
    upsample_signal = np.zeros((N_ch * Naz, Nrg), dtype=complex)
    for k in range(Naz):
        upsample_signal[5*k:5*k+5, :] = reconstr_signal[:, k, :]

    upsample_signal = N_ch * np.fft.ifft(np.fft.ifftshift(upsample_signal, axes=0), axis=0)  # IFFT to get back to time domain
    # add the dimension of polarization
    upsample_signal = upsample_signal[None, :, :]
    print('Upsampling completed!') 
    # save the recontructed raw data to a new file
    # better to put together with the raw_data to reduce the volume in the future
    reconstr_file = tpio.ReconstructedRawFile(reconstr_output_file, 'w', upsample_signal.shape)
    reconstr_file.set('inc_angle', np.rad2deg(inc_angle))
    reconstr_file.set('f0', f0)
    reconstr_file.set('ant_L', ant_L)
    reconstr_file.set('prf', prf)
    reconstr_file.set('v_ground', v_ground)
    reconstr_file.set('az0', az0)
    reconstr_file.set('orbit_alt', alt)
    reconstr_file.set('sr0', sr0)
    reconstr_file.set('rg_sampling', rg_sampling)
    reconstr_file.set('rg_bw', rg_bw)
    reconstr_file.set('raw_data*', upsample_signal)
    reconstr_file.close()