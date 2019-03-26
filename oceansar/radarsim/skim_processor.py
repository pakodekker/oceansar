#!/usr/bin/env python

""" ============================================
    SRP: SAR Raw data Processor (:mod:`srp`)
    ============================================

    Script to process SAR RAW data

    **Arguments**
        * -c, --cfg_file: Configuration file
        * -r, --raw_file: Raw data file
        * -o, --output_file: Output file

"""

import os
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

from oceansar.utils import geometry as geo
from oceansar import utils
from oceansar import ocs_io as tpio
from oceansar import constants as const


def range_spectrum(data, fs, pdir):
    # Basically, to check range spectrum of data
    spec = np.fft.fft(data, axis=1)
    spec = np.mean(np.abs(spec)**2, axis=0)
    freq = np.fft.fftfreq(spec.size, 1/fs)
    plt.figure()
    plt.plot(np.fft.fftshift(freq)/1e6, np.fft.fftshift(spec))
    plt.xlabel("f [MHz]")
    plt.grid(True)
    plt.savefig(os.path.join(pdir, 'fastfreq_spectrum'))


def range_oversample(data, n=2):
    data_f = np.fft.fft(data, axis=1)
    data_ovs_f = np.zeros((data.shape[0], n * data.shape[1]), dtype=np.complex)
    data_ovs_f[:, 0:data.shape[1]] = np.fft.fftshift(data_f, (1,))
    data_ovs_f = np.roll(data_ovs_f, -int(data.shape[1] / 2), axis=1)
    data_ovs = np.fft.ifft(data_ovs_f, axis=1)
    return data_ovs


def pulse_pair(data, prf):
    # stricktly speaking, we should oversample, so let's do it

    pp = data[1:] * np.conj(data[0:-1])
    data_int = np.abs(data)
    mean_intensity_profile = np.mean(data_int, axis=0)
    mask = mean_intensity_profile > (mean_intensity_profile.mean() / 10)
    mask = mask.reshape((1, mask.size))
    # Phase to dop
    p2d = 1 / (2 * np.pi) * prf
    # average complex phasors
    pp_rg_avg = np.mean(pp, axis=-1)
    pp_rg_avg_dop = p2d * np.angle(pp_rg_avg)
    # average phase to eliminate biases due to amplitude variations
    # FIXME
    ppm = pp * mask
    good = np.where(np.abs(ppm) > 0)
    ppm[good] = ppm[good]/np.abs(ppm[good])
    # phase_rg_avg_dop = p2d * np.mean(np.angle(pp[:, :]), axis=-1)
    phase_rg_avg_dop = p2d * np.angle(np.mean(ppm[:, :], axis=-1))
    # Coherence
    coh_rg_avg = pp_rg_avg / np.sqrt(np.mean(np.abs(data[1:])**2, axis=-1) *
                                     np.mean(np.abs(data[0:-1])**2, axis=-1))

    return pp_rg_avg_dop, phase_rg_avg_dop, np.abs(coh_rg_avg)


def rar_spectra(data, fs, rgsmth=4):
    nrg = data.shape[1]
    pp = data[1:] * np.conj(data[0:-1])
    # Intensity spectrum
    data_int = utils.smooth(np.abs(data) ** 2, 10, axis=0)
    mean_intensity_profile = np.mean(data_int, axis=0)
    mask = mean_intensity_profile > (mean_intensity_profile.mean() / 10)
    mask = mask.reshape((1, mask.size))
    data_int_m = mask * data_int
    mean_int = np.sum(data_int_m) / np.sum(mask) / data.shape[0]
    mean_intensity_profile2 = mean_intensity_profile - mean_int * mask.flatten()
    han = np.hanning(data.shape[1]).reshape((1, data.shape[1]))
    data_ac = data_int_m - mask * mean_int
    intens_spectrum = utils.smooth(np.mean(np.abs(np.fft.fft(han * data_ac, axis=1)) ** 2, axis=0), rgsmth)/nrg
    phase_spectrum = np.mean(np.abs(np.fft.fft(mask * np.angle(utils.smooth(pp, 10, axis=0)), axis=1)) ** 2, axis=0)
    phase_spectrum = utils.smooth(phase_spectrum, rgsmth)/nrg
    kr = 2 * np.pi * np.fft.fftfreq(intens_spectrum.size, const.c / 2 / fs)
    return kr, mean_intensity_profile2, intens_spectrum, phase_spectrum


def sar_spectra(sardata, fs, rgsmth=4):
    # azimuth int profile
    nrg = sardata.shape[1]
    az_prof = np.mean(sardata, axis=1)
    mask = sardata > (az_prof.reshape((az_prof.size, 1)) / 10)
    az_prof = np.sum(sardata, axis=1) / np.sum(mask, axis=1)
    # Remove average
    han = np.hanning(sardata.shape[1]).reshape((1, sardata.shape[1]))
    sardata_ac = mask * (sardata - az_prof.reshape((az_prof.size, 1)))
    sarint_spec = np.mean(np.abs(np.fft.fft(sardata_ac * han, axis=1))**2, axis=0)/nrg
    sarint_spec = utils.smooth(sarint_spec, rgsmth)
    kr = 2 * np.pi * np.fft.fftfreq(sarint_spec.size, const.c / 2 / fs)
    return kr, sarint_spec


def unfocused_sar(data, n_sar):
    dimsin = data.shape
    if np.size(dimsin) == 1:
        data_rshp = data.reshape((int(dimsin[0] / n_sar), int(n_sar)))
    else:
        data_rshp = data.reshape((int(dimsin[0] / n_sar), int(n_sar), dimsin[1]))
    # DFT in azimuth
    data_ufcs = np.fft.fft(data_rshp, axis=1)/np.sqrt(n_sar)
    # focused average intensity
    int_ufcs = np.mean(np.abs(data_ufcs)**2, axis=0)
    return int_ufcs, data_ufcs


def sar_delta_k_slow(sar_data, fs, dksmoth=4):
    # Number of range samples
    nrg = sar_data.shape[2]
    ndk = int(nrg/2)
    # FFT in range and re-order
    sar_data_f = np.fft.fftshift(np.fft.fft(sar_data, axis=2), axes=(2,))
    dk_sig = np.zeros((sar_data.shape[0], sar_data.shape[1], ndk), dtype=np.complex)
    for ind in range(0, ndk):
        dk_ind = sar_data_f[:, :, ind:] * np.conj(sar_data_f[:, :, 0:nrg - ind])
        dk_sig[:, :, ind] = np.mean(dk_ind, axis=2)
    dk_avg = utils.smooth(np.mean(np.abs(np.mean(dk_sig, axis=0))**2, axis=0), dksmoth)
    # dkr = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ndk, const.c / 2 / fs))
    dkr = fs / nrg * 2 / const.c * 2 * np.pi * np.arange(ndk)
    return dkr, dk_avg, dk_sig


def sar_delta_k(csardata, fs, dksmoth=4):
    # Number of range samples
    nrg = csardata.shape[2]
    sardata = np.abs(csardata)**2
    az_prof = np.mean(np.mean(sardata, axis=2), axis=0)
    burst_avg = np.mean(sardata, axis=0)
    mask = burst_avg > (az_prof.reshape((az_prof.size, 1)) / 10)
    mask = mask.reshape((1,) + mask.shape)
    # Remove average
    han = np.hanning(sardata.shape[2]).reshape((1, 1, sardata.shape[2]))
    sardata_ac = mask * (sardata - az_prof.reshape((1, az_prof.size, 1)))
    dk_sig = np.fft.fft(sardata_ac * han, axis=2) / nrg
    # Keep only positive frequencies, since input signal was real
    ndk = int(nrg/2)
    dk_sig = dk_sig[:, :, 0:int(ndk)]
    dk_avg = utils.smooth(np.mean(np.abs(np.mean(dk_sig, axis=0))**2, axis=0), dksmoth)
    # After smoothing we would down-sample, but not needed  for simulator
    dkr = fs / nrg * 2 / const.c * 2 * np.pi * np.arange(ndk)
    return dkr, dk_avg, dk_sig


def sar_delta_k_rcm(dkr, dk_sig, drcm_dt, inter_aperture_time):
    dims = dk_sig.shape
    tv = np.arange(dims[0]).reshape((dims[0], 1, 1)) * inter_aperture_time
    dkr_rshp = dkr.reshape((1, 1, dims[2]))
    dphase = tv * dkr_rshp * drcm_dt.reshape((1, dims[1], 1))
    return dk_sig * np.exp(-1j * dphase)


def sar_delta_k_omega(dk_sig, inter_aperture_time, dksmoth=4):
    n_block = dk_sig.shape[0]
    dk_pps = np.zeros((n_block-1, dk_sig.shape[2]), dtype=np.complex)
    dk_omega = np.zeros_like(dk_pps, dtype=np.float)
    for ind_lag in range(1, n_block):
        dk_pp = dk_sig[ind_lag:, :, :] * np.conj(dk_sig[0:n_block - ind_lag, :, :])
        dk_pps[ind_lag-1] = np.mean(np.mean(dk_pp, axis=0), axis=0)
        # dk_pps[ind_lag - 1] = np.mean(dk_pp, axis=0)[0]
        # Angular frequencies
        dk_omega[ind_lag-1] = np.angle(utils.smooth(dk_pps[ind_lag-1], dksmoth)) / (inter_aperture_time * ind_lag)

    #d k_omega = utils.smooth(dk_omega, dksmoth, axis=1)
    return dk_pps, dk_omega



def skim_process(cfg_file, raw_output_file):

    ###################
    # INITIALIZATIONS #
    ###################


    # CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)
    info = utils.PrInfo(cfg.sim.verbosity, "processor")
    # Say hello
    info.msg(time.strftime("Starting: %Y-%m-%d %H:%M:%S", time.localtime()))
    # PROCESSING
    az_weighting = cfg.processing.az_weighting
    doppler_bw = cfg.processing.doppler_bw
    plot_format = cfg.processing.plot_format
    plot_tex = cfg.processing.plot_tex
    plot_save = cfg.processing.plot_save
    plot_path = cfg.processing.plot_path
    plot_raw = cfg.processing.plot_raw
    plot_rcmc_dopp = cfg.processing.plot_rcmc_dopp
    plot_rcmc_time = cfg.processing.plot_rcmc_time
    plot_image_valid = cfg.processing.plot_image_valid
    doppler_demod = cfg.processing.doppler_demod
    pp_file = os.path.join(cfg.sim.path, cfg.sim.pp_file)

    # radar
    f0 = cfg.radar.f0
    prf = cfg.radar.prf
    # num_ch = cfg.radar.num_ch
    alt = cfg.radar.alt
    v_ground = cfg.radar.v_ground

    # CALCULATE PARAMETERS
    l0 = const.c / f0
    if v_ground == 'auto':
        v_ground = geo.orbit_to_vel(alt, ground=True)
    rg_sampling = cfg.radar.Fs
    # Range freqency axis
    f_axis = np.linspace(0, rg_sampling, cfg.radar.n_rg)
    wavenum_scale = f_axis * 4 * np.pi / const.c * np.sin(np.radians(cfg.radar.inc_angle))

    # RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    info.msg("Raw data max: %f" % (np.max(np.abs(raw_data))))
    dop_ref = raw_file.get('dop_ref')
    sr0 = raw_file.get('sr0')
    azimuth = raw_file.get('azimuth')
    raw_file.close()
    #n_az

    # OTHER INITIALIZATIONS
    # Create plots directory
    plot_path = os.path.dirname(pp_file) + os.sep + plot_path
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    ########################
    # PROCESSING MAIN LOOP #
    ########################

    # Optimize matrix sizes
    az_size_orig, rg_size_orig = raw_data[0].shape
    optsize = utils.optimize_fftsize(raw_data[0].shape)
    # optsize = [optsize[0], optsize[1]]
    data = np.zeros(optsize, dtype=complex)
    data[0:az_size_orig, 0:rg_size_orig] = raw_data[0]
    # Doppler demodulation according to geometric Doppler
    if doppler_demod:
        info.msg("Doppler demodulation")
        t_vec = (np.arange(optsize[0])/prf).reshape((optsize[0], 1))
        data[:, 0:rg_size_orig] = (data[:, 0:rg_size_orig] *
                                   np.exp((-2j * np.pi) * t_vec * dop_ref.reshape((1, rg_size_orig))))
    # Pulse pair
    info.msg("Range over-sampling")
    data_ovs = range_oversample(data)
    info.msg("Pulse-pair processing")
    dop_pp_avg, dop_pha_avg, coh = pulse_pair(data_ovs[0:az_size_orig, 0:2 * rg_size_orig], prf)
    krv, mean_int_profile, int_spe, phase_spec = rar_spectra(data_ovs[0:az_size_orig, 0:2 * rg_size_orig],
                                                             2 * rg_sampling, rgsmth=8)
    kxv = krv * np.sin(np.radians(cfg.radar.inc_angle))
    range_spectrum(data[0:az_size_orig, 0:rg_size_orig], rg_sampling, plot_path)
    info.msg("Mean DCA (pulse-pair average): %f Hz" % (np.mean(dop_pp_avg)))
    info.msg("Mean DCA (pulse-pair phase average): %f Hz" % (np.mean(dop_pha_avg)))
    info.msg("Mean coherence: %f " % (np.mean(coh)))

    # Ground range projection
    if cfg.processing.ground_project:
        info.msg("Projecting to ground range")
        # Slant range vector
        sr = (np.arange(2 * rg_size_orig) - rg_size_orig) * const.c / rg_sampling / 2 + sr0
        gr, theta_i_v, theta_l_v, b_v = geo.sr_to_geo(sr, alt)
        gr_out = np.linspace(gr.min(), gr.max(), 2 * rg_size_orig)
        sr_out, theta_i_v, theta_l_v = geo.gr_to_geo(gr_out, alt)
        ind_sr_out = (sr_out - sr[0]) / (sr[1] - sr[0]) * 4
        data_ovs = range_oversample(data, n=8)
        data_ovs = utils.linresample(data_ovs, ind_sr_out, axis=1, extrapolate=True)

    # Unfocused SAR
    info.msg("Unfocused SAR")
    int_unfcs, data_ufcs = unfocused_sar(data_ovs[0:az_size_orig, 0:2*rg_size_orig], cfg.processing.n_sar)
    # Doppler vector
    dop_bins = np.fft.fftfreq(cfg.processing.n_sar, 1/prf)
    # Range cell migration rate
    dresrcm_dt = - dop_bins * l0 /2
    krv2, sar_int_spec = sar_spectra(int_unfcs, 2 * rg_sampling, rgsmth=8)

    # Some delta-k on focused sar data
    info.msg("Unfocused SAR delta-k spectrum")
    dkr, dk_avg, dk_signal = sar_delta_k(data_ufcs, 2 * rg_sampling, dksmoth=8)
    if cfg.processing.rcm:
        info.msg("Applying RCM to delta-k signal phase")
        dk_signal = sar_delta_k_rcm(dkr, dk_signal, dresrcm_dt, cfg.processing.n_sar/prf)
        
    dk_pulse_pairs, dk_omega = sar_delta_k_omega(dk_signal, cfg.processing.n_sar/prf, dksmoth=16)
    # For verification, comment out
    # dkr_sl, dk_avg_sl, dk_signal_sl = sar_delta_k_slow(data_ufcs, 2 * rg_sampling, dksmoth=8)
    # dk_pulse_pairs_sl, dk_omega_sl = sar_delta_k_omega(dk_signal_sl, cfg.processing.n_sar/prf, dksmoth=32)
    info.msg(time.strftime("Processing done [%Y-%m-%d %H:%M:%S]", time.localtime()))

    if plot_raw:
        plt.figure()
        plt.imshow(np.real(raw_data[0]), vmin=-np.max(np.abs(raw_data[0])), vmax=np.max(np.abs(raw_data[0])),
                   origin='lower', aspect=np.float(raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                   cmap='viridis')
        #plt.title()
        plt.xlabel("Range [samples]")
        plt.ylabel("Azimuth [samples")
        plt.savefig(plot_path + os.sep + 'plot_raw_real.%s' % plot_format, dpi=150)
        plt.close()
        plt.figure()
        plt.imshow(np.abs(raw_data[0]), vmin=0, vmax=np.max(np.abs(raw_data[0])),
                   origin='lower', aspect=np.float(raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                   cmap='viridis')
        #plt.title()
        plt.xlabel("Range [samples]")
        plt.ylabel("Azimuth [samples")
        plt.savefig(plot_path + os.sep + 'plot_raw_abs.%s' % plot_format, dpi=150)
        plt.close()

    plt.figure()
    plt.imshow(np.fft.fftshift(int_unfcs, axes=(0,)),
               origin='lower', aspect='auto',
               cmap='viridis')
    # plt.title()
    plt.xlabel("Range [samples]")
    plt.ylabel("Doppler [samples")
    plt.savefig(plot_path + os.sep + 'ufcs_int.%s' % plot_format, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(mean_int_profile)
    plt.xlabel("Range samples [Pixels]")
    plt.ylabel("Intensity")
    plt.savefig(plot_path + os.sep + 'mean_int.%s' % plot_format)
    plt.close()

    plt.figure()
    plt.plot(np.fft.fftshift(kxv), np.fft.fftshift(int_spe))
    plt.ylim((int_spe[20:np.int(cfg.radar.n_rg)-20].min()/2,
              int_spe[20:np.int(cfg.radar.n_rg)-20].max()*1.5))
    plt.xlim((0, kxv.max()))
    plt.xlabel("$k_x$ [rad/m]")
    plt.ylabel("$S_I$")
    plt.savefig(plot_path + os.sep + 'int_spec.%s' % plot_format)
    plt.close()

    plt.figure()
    plt.plot(np.fft.fftshift(kxv), np.fft.fftshift(sar_int_spec))
    plt.ylim((sar_int_spec[20:np.int(cfg.radar.n_rg)-20].min()/2,
              sar_int_spec[20:np.int(cfg.radar.n_rg)-20].max()*1.5))
    plt.xlim((0, kxv.max()))
    plt.xlabel("$k_x$ [rad/m]")
    plt.ylabel("$S_I$")
    plt.savefig(plot_path + os.sep + 'sar_int_spec.%s' % plot_format)
    plt.close()

    plt.figure()
    plt.plot(np.fft.fftshift(kxv), np.fft.fftshift(phase_spec))
    plt.ylim((phase_spec[20:np.int(cfg.radar.n_rg)-20].min()/2,
              phase_spec[20:np.int(cfg.radar.n_rg)-20].max()*1.5))
    plt.xlabel("$k_x$ [rad/m]")
    plt.xlim((0, kxv.max()))
    plt.ylabel("$S_{Doppler}$")
    plt.savefig(plot_path + os.sep + 'pp_phase_spec.%s' % plot_format)

    plt.figure()
    dkx = dkr * np.sin(np.radians(cfg.radar.inc_angle))
    plt.plot(dkx, dk_avg)
    plt.ylim((dk_avg[20:dkx.size-20].min()/2,
              dk_avg[20:dkx.size-20].max()*1.5))
    plt.xlim((0.1, dkx.max()))
    plt.xlabel("$\Delta k_x$ [rad/m]")
    plt.ylabel("$S_I$")
    plt.savefig(plot_path + os.sep + 'sar_delta_k_spec.%s' % plot_format)
    plt.close()

    # plt.figure()
    # dkx_sl = dkr_sl * np.sin(np.radians(cfg.radar.inc_angle))
    # plt.plot(dkx_sl, dk_avg_sl)
    # plt.ylim((dk_avg_sl[20:dkx.size-20].min()/2,
    #           dk_avg_sl[20:dkx.size-20].max()*1.5))
    # plt.xlim((0, dkx.max()))
    # plt.xlabel("$\Delta k_x$ [rad/m]")
    # plt.ylabel("$S_I$")
    # plt.savefig(plot_path + os.sep + 'sar_delta_k_spec_slow.%s' % plot_format)
    # plt.close()

    plt.figure()
    # plt.plot(dkx, dk_omega[0], label="lag=1")
    plt.plot(dkx, dk_omega[1], label="lag=2")
    # plt.plot(dkx, dk_omega_sl[0] + 1, label="slow-lag=1")
    # plt.plot(dkx, dk_omega_sl[1] + 1, label="slow-lag=2")
    plt.plot(dkx, dk_omega[3], label="lag=4")
    # plt.plot(dkx, dk_omega[-1], label="lag=max")
    plt.ylim((-20, 20))
    #plt.ylim((dk_avg[20:dkx.size - 20].min() / 2,
    #          dk_avg[20:dkx.size - 20].max() * 1.5))
    plt.xlim((0.05, 0.8))
    plt.xlabel("$\Delta k_x$ [rad/m]")
    plt.ylabel("$\omega$ [rad/s]")
    plt.legend(loc=0)
    plt.savefig(plot_path + os.sep + 'sar_delta_k_omega.%s' % plot_format)
    plt.close()

    info.msg("Saving output to %s" % pp_file)
    np.savez(pp_file,
             dop_pp_avg=dop_pp_avg,
             dop_pha_avg=dop_pha_avg,
             coh=coh,
             ufcs_intensity=int_unfcs,
             mean_int_profile=mean_int_profile,
             int_spec=int_spe, sar_int_spec=sar_int_spec,
             ppphase_spec=phase_spec, kx=kxv,
             dk_spec=dk_avg,
             dk_omega=dk_omega,
             dkx=dkx)

    info.msg(time.strftime("All done [%Y-%m-%d %H:%M:%S]", time.localtime()))


def raw_data_extraction(raw_output_file=None):

    ###################
    # INITIALIZATIONS #
    ###################
    # RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    raw_file.close()
    return raw_data


def delta_k_processing(raw_output_file, cfg_file):
    # heading
    # flush is true is for correctly showing
    print('-------------------------------------------------------------------', flush=True)
    print('Delta-k processing begins...')
    # parameters
    cfg = tpio.ConfigFile(cfg_file)
    pp_file = os.path.join(cfg.sim.path, cfg.sim.pp_file)
    n_sar_a = cfg.processing.n_sar
    Azi_img = cfg.processing.Azi_img
    # radar
    f0 = cfg.radar.f0
    prf = cfg.radar.prf
    # num_ch = cfg.radar.num_ch
    alt = cfg.radar.alt
    v_ground = cfg.radar.v_ground
    wei_dop = 0
    m_ind = 0

    # CALCULATE PARAMETERS
    l0 = const.c / f0
    if v_ground == 'auto':
        v_ground = geo.orbit_to_vel(alt, ground=True)

    if Azi_img:  # 2-D unfocusing
        data = np.load(pp_file)
        Doppler_av = np.mean(data['dop_pp_avg'])
        dp_axis = np.linspace(-prf / 2, prf / 2, n_sar_a)
        val_d = np.abs(dp_axis - Doppler_av)
        m_d = np.where(val_d == min(val_d))
        m_ind = np.int(m_d[0])
        f_in = prf / n_sar_a
        beamwide = l0 / cfg.radar.ant_L
        Bw = 2 * v_ground / l0 * beamwide
        Bw_eff = np.abs(Bw * np.sin(cfg.radar.azimuth))
        wei_dop = Bw_eff / f_in

    # Analyse different waves
    sim_path_ref = cfg.sim.path + os.sep + 'wavelength%.1f'
    for inn in range(np.size(cfg.processing.wave_scale)):
        wave_scale = cfg.processing.wave_scale[inn]
        path_p = sim_path_ref % (wave_scale)
        if not (os.path.exists(path_p) | Azi_img):
            os.makedirs(path_p)
        else:
            sim_path_ref = cfg.sim.path + os.sep + 'wavelength%.1f_unfocus'
            path_p = sim_path_ref % (wave_scale)
            if not os.path.exists(path_p):
                os.makedirs(path_p)

        # imaging signs
        plot_pattern = False
        plot_spectrum = False

        # processing parameters and initial parameters
        # radar parameters
        Az_smaples = cfg.radar.n_pulses
        PRF = cfg.radar.prf
        fs = cfg.radar.Fs
        R_samples = cfg.radar.n_rg
        inc = cfg.radar.inc_angle
        n_sar_r = cfg.processing.n_sar_r
        R_n = round(cfg.ocean.Lx * cfg.ocean.dx * np.sin(inc*np.pi/180) / (const.c / 2 / fs))
        rang_img = cfg.processing.rang_img
        analysis_deltan = np.linspace(0, 500, 501)  # for delta-k spectrum analysis
        r_int_num = cfg.processing.r_int_num
        az_int = cfg.processing.az_int
        Delta_lag = cfg.processing.Delta_lag
        num_az = cfg.processing.num_az
        list = range(1, num_az)
        analyse_num = range(2, Az_smaples - az_int - 1 - num_az)
        Scene_scope = ((R_samples - 1) * const.c / fs / 2) / 2
        k_w = 2 * np.pi / wave_scale
        # initialization parameters
        dk_higha = np.zeros((np.size(analyse_num),r_int_num), dtype=np.float)
        # RCS_power = np.zeros_like(analysis_deltan)

        # get raw data
        raw_data = raw_data_extraction(raw_output_file)
        raw_data = raw_data[0]

        # showing pattern of the ocean waves
        # intensity
        raw_int = raw_data * np.conj(raw_data)
        if plot_pattern:
            plt.figure()
            plt.imshow(np.abs(raw_int))
            plt.xlabel("Range (pixel)")
            plt.ylabel("Azimuth (pixel)")

            plot_path = cfg.sim.path + os.sep + 'delta_k_spectrum_plots'

            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            plt.savefig(os.path.join(plot_path, 'Pattern.png'))
            plt.close()
        intens_spectrum = np.mean(np.fft.fft(np.abs(raw_int), axis=1), axis=0)
        if plot_spectrum:
            plt.figure()
            plt.plot(10 * np.log10(np.abs(intens_spectrum[1:np.int(R_samples / 2)])))
            plt.xlabel("Delta_k [1/m]")
            plt.ylabel("Power (dB)")

        # Showing Delta-k spectrum for analysis
        delta_k_spectrum(Scene_scope, r_int_num, inc,
                         raw_data, analysis_deltan, path_p)
        # Calculating the required delta_f value
        delta_f = cal_delta_f(inc, wave_scale)
        delta_k = delta_f / const.c
        ind_N = np.int(round(delta_k * (2 * Scene_scope) * 2))
        # sar processing parameters transfor
        s_r, r_int_num, ind_N, data_rshp, s_a, az_int, analyse_num, n_sar_a = Transform(raw_data, R_samples,
                    n_sar_r, r_int_num, Az_smaples, ind_N, az_int, n_sar_a, analyse_num, rang_img, Azi_img)
        # initialization parameters
        dk_higha = np.zeros((np.size(analyse_num), r_int_num), dtype = np.float)
        #  = np.zeros_like(analysis_deltan)
        Omiga_p = np.zeros((np.size(analyse_num), np.size(list) + 1), dtype=np.float)
        Omiga_p_z = np.zeros(np.size(analyse_num), dtype = np.float)
        Phase_p = np.zeros((np.size(analyse_num),np.size(list)+1), dtype=np.float)
        Phase_p_z = np.zeros(np.size(analyse_num), dtype=np.float)

        # Delta-k processing
        if rang_img & Azi_img:  # 2-D unfocusing
            for ind_x in range(s_r):
                r_data = data_rshp[:, ind_x, :]
                spck_f = np.fft.fftshift(np.fft.fft(r_data, axis=1), axes=(1,))
                dk_high = spck_f[:, 0:r_int_num] * np.conj(spck_f[:, ind_N:ind_N + r_int_num])
                dk_higha = np.mean(dk_high, axis=1)
                # azimuth sar azimuth processing
                dimsin = dk_higha.shape
                int_unfcs, data_a = unfocused_sar(dk_higha, n_sar_a)
                data_ufcs = np.fft.fftshift(data_a, axes=(1,))

                for ind in range(np.size(analyse_num)):
                   (Omiga_p_z[ind],
                    Phase_p_z[ind]) = Cal_pha_vel(data_ufcs[analyse_num[ind]:
                                                            analyse_num[ind]+az_int,:],
                                                  data_ufcs[0:az_int,:],
                                                  analyse_num[ind], PRF, k_w,
                                                  n_sar_a, m_ind, wei_dop, Azi_img)
                Omiga_p_z[ind] = Omiga_p_z[ind] + Omiga_p_z[ind]
                Phase_p_z[ind] = Phase_p_z[ind] + Phase_p_z[ind]
            Omiga_p_z = Omiga_p_z / s_r
            Phase_p_z = Phase_p_z / s_r

        elif rang_img: # range unfocusing
            for ind_x in range(s_r):
                r_data = data_rshp[:, ind_x, :]
                spck_f = np.fft.fftshift(np.fft.fft(r_data, axis=1), axes=(1,))
                dk_high = spck_f[:, 0:r_int_num]*np.conj(spck_f[:, ind_N:ind_N+r_int_num])
                dk_higha = np.mean(dk_high, axis=1)
                for ind in range(np.size(analyse_num)):
                    (Omiga_p_z[ind],
                     Phase_p_z[ind]) = Cal_pha_vel(dk_higha[analyse_num[ind]:
                                                            analyse_num[ind]+az_int],
                                                   dk_higha[0:az_int],
                                                   analyse_num[ind], PRF, k_w,
                                                   n_sar_a, m_ind, wei_dop, Azi_img)
                Omiga_p_z[ind] = Omiga_p_z[ind] + Omiga_p_z[ind]
                Phase_p_z[ind] = Phase_p_z[ind] + Phase_p_z[ind]
            Omiga_p_z = Omiga_p_z / s_r
            Phase_p_z = Phase_p_z / s_r
        elif Azi_img: # azimuth unfocusing
            spck_f = np.fft.fftshift(np.fft.fft(raw_data, axis=1), axes=(1,))
            dk_high = spck_f[:,0:r_int_num]*np.conj(spck_f[:,ind_N:ind_N+r_int_num])
            dk_higha = np.mean(dk_high, axis=1)
            # azimuth sar azimuth processing
            int_unfcs, data_a = unfocused_sar(dk_higha, n_sar_a)
            data_ufcs = np.fft.fftshift(data_a, axes=(1,))

            # Estimate phase
            for ind in range(np.size(analyse_num)):
                (Omiga_p_z[ind],
                 Phase_p_z[ind]) = Cal_pha_vel(data_ufcs[analyse_num[ind]:
                                                         analyse_num[ind]+az_int,:],
                                               data_ufcs[0:az_int,:],
                                               analyse_num[ind], PRF, k_w,
                                               n_sar_a, m_ind, wei_dop, Azi_img)
        # Considering delta-k processing with lags
        elif Delta_lag:
            spck_f = np.fft.fftshift(np.fft.fft(raw_data, axis=1), axes=(1,))
            for iii in list:
                for ind in range(np.size(analyse_num)):
                    if iii == 0:
                        dk_inda = spck_f[iii:, 0:r_int_num] * np.conj(spck_f[0:, ind_N:ind_N + r_int_num])
                    else:
                        dk_inda = spck_f[iii:, 0:r_int_num] * np.conj(spck_f[0:-iii, ind_N:ind_N + r_int_num])
                    dk_higha = np.mean(dk_inda, axis=1)
                    (Omiga_p[ind, iii],
                     Phase_p[ind, iii]) = Cal_pha_vel(dk_higha[analyse_num[ind]:
                                                               analyse_num[ind]+az_int],
                                                      dk_higha[0:az_int],
                                                      analyse_num[ind], PRF, k_w,
                                                      n_sar_a, m_ind, wei_dop, Azi_img)

        else:  # none unfocusing
            spck_f = np.fft.fftshift(np.fft.fft(raw_data, axis=1), axes=(1,))

            # Extracting the information of wave_scale
            dk_high = spck_f[:, 0:r_int_num] * np.conj(spck_f[:, ind_N:ind_N+r_int_num])
            dk_higha = np.mean(dk_high, axis=1)
            for ind in range(np.size(analyse_num)):
                (Omiga_p_z[ind],
                 Phase_p_z[ind]) = Cal_pha_vel(dk_higha[analyse_num[ind]:
                                                        analyse_num[ind]+az_int],
                                               dk_higha[0:az_int],
                                               analyse_num[ind], PRF, k_w,
                                               n_sar_a, m_ind, wei_dop, Azi_img)

        # processing results
        if Azi_img:
            analyse_num = np.array(analyse_num) * n_sar_a
        plt.figure()
        plt.plot(analyse_num, Omiga_p_z)
        plt.xlabel("Azimuth interval [Pixel]")
        plt.ylabel("Angular velocity (rad/s)")

        plot_path = path_p + os.sep + 'delta_k_spectrum_plots'
        if not os.path.exists(path_p):
            os.makedirs(path_p)
        plt.savefig(os.path.join(plot_path, 'Angular_velocity.png'))
        plt.close()

        plt.figure()
        plt.plot(analyse_num, Phase_p_z)
        plt.xlabel("Azimuth interval [Pixel]")
        plt.ylabel("Phase velocity (m/s)")

        plt.savefig(os.path.join(plot_path, 'Phase_velocity.png'))
        plt.close()
        # save the result data
        if Delta_lag:  # lag case
            Omiga_f = Omiga_p[:, 0]
            Phase_f = Phase_p[:, 0]
        else:  # normal case
            Omiga_f = Omiga_p_z
            Phase_f = Phase_p_z
        Omiga_s = Omiga_f
        Phase_s = Phase_f

        np.save(os.path.join(plot_path, 'Angular_velocity.npy'),
                [Omiga_s, np.mean(Omiga_s), np.size(Omiga_s), analyse_num])

        np.save(os.path.join(plot_path, 'Phase_velocity.npy'),
                [Phase_s, np.mean(Phase_s), np.size(Phase_s), analyse_num])


def cal_delta_f(inc, wave_scale):

    delta_ff = const.c / 2 / wave_scale / np.sin(inc * np.pi / 180)

    return delta_ff


def delta_k_spectrum(scene_scope, r_int_num, inc,
                     raw_data,
                     analysis_deltan, path_p):

    spck_f = np.fft.fftshift(np.fft.fft(raw_data, axis=1), axes=(1,))

    analysis_deltaf = analysis_deltan / 4 / scene_scope * const.c
    nrg = raw_data.shape[1]
    rcs_power = np.zeros(np.size(analysis_deltan))
    for ind in range(np.size(analysis_deltan)):
        dk_ind = spck_f[:, ind:] * np.conj(spck_f[:, 0:nrg-ind])
        rcs_power[ind] = np.abs(np.mean(dk_ind))

    xx = analysis_deltaf[1:np.size(analysis_deltan)] / 1e6
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(xx, 10*(np.log10(rcs_power[1:np.size(analysis_deltan)])))
    ax1.set_xlabel("Delta_f [MHz]")
    ax1.set_ylabel("Power (dB)")
    plot_path = path_p + os.sep + 'delta_k_spectrum_plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plt.savefig(os.path.join(plot_path, 'delta_k_spectrum.png'))
    plt.close()


def Cal_pha_vel(data1, data2, in_no, PRF, k_w,
                n_sar_a, m_ind, wei_dop=0, Azi_img=False):
    if Azi_img:
        pha_f = -np.angle(np.mean(data1 * np.conj(data2), axis=0))
        pha_ff = pha_f [int(m_ind - round(wei_dop / 2)):int(m_ind + round(wei_dop / 2))]
        pha = np.mean(pha_ff)
        Omiga = pha / in_no * PRF / n_sar_a
        Phase = Omiga / k_w
    else:
        pha = -np.angle(np.mean(data1 * np.conj(data2)))
        Omiga = pha / in_no * PRF / n_sar_a
        Phase = Omiga / k_w
    return Omiga, Phase


def Transform(raw_data, R_samples, n_sar_r, r_int_num,
              Az_smaples, ind_N, az_int, n_sar_a, analyse_num,
              rang_img, Azi_img):
    if rang_img & Azi_img:
        s_rr = round(R_samples / n_sar_r)
        r_int_num = round(r_int_num/s_rr)
        ind_N = round(ind_N/s_rr)
        data_rshp = raw_data.reshape((Az_smaples, s_rr, n_sar_r))
        s_a = round(Az_smaples/n_sar_a)
        az_int = int(az_int / n_sar_a)-2
        analyse_num = range(2, s_a - az_int - 1)
    elif rang_img:
        s_rr = round(R_samples / n_sar_r)
        r_int_num = round(r_int_num/s_rr)
        ind_N = round(ind_N/s_rr)
        data_rshp = raw_data.reshape((Az_smaples, s_rr, n_sar_r))
        s_a = Az_smaples
    elif Azi_img:
        s_rr = R_samples
        s_a = round(Az_smaples/n_sar_a)
        az_int = int(az_int / n_sar_a)-2
        analyse_num = range(2, s_a - az_int - 1)
        data_rshp = raw_data
    else:
        s_rr = R_samples
        n_sar_a = 1
        data_rshp = raw_data
        s_a = Az_smaples
    return s_rr, r_int_num, ind_N, data_rshp, s_a, az_int, analyse_num, n_sar_a



if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-r', '--raw_file')
    # parser.add_argument('-o', '--output_file')
    args = parser.parse_args()
    extra_dk_proc = False
    if extra_dk_proc:
        skim_process(args.cfg_file, args.raw_file)
        delta_k_processing(args.raw_file, args.cfg_file)
    else:
        skim_process(args.cfg_file, args.raw_file)
        # delta_k_processing(args.raw_file, args.cfg_file)
