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





def pulse_pair(data, prf):
    pp = data[1:] * np.conj(data[0:-1])
    # Phase to dop
    p2d = 1 / (2 * np.pi) * prf
    # average complex phasors
    pp_rg_avg = np.mean(pp, axis=-1)
    pp_rg_avg_dop = p2d * np.angle(pp_rg_avg)
    # average phase to eliminate biases due to amplitude variations
    # FIXME
    phase_rg_avg_dop = p2d * np.mean(np.angle(pp[:, 700:1300]), axis=-1)
    # Coherence
    coh_rg_avg = pp_rg_avg / np.sqrt(np.mean(np.abs(data[1:])**2, axis=-1) *
                                     np.mean(np.abs(data[0:-1])**2, axis=-1))
    # Intensity spectrum
    data_int = np.abs(data) ** 2
    mean_intensity_profile = np.mean(np.abs(data_int), axis=0)
    intens_spectrum = np.mean(np.fft.fft(np.abs(data_int), axis=1), axis=0)
    intens_spectrum_smth = utils.smooth(intens_spectrum, 20)
    return (pp_rg_avg_dop, phase_rg_avg_dop, np.abs(coh_rg_avg),
            mean_intensity_profile, intens_spectrum, intens_spectrum_smth)


def unfocused_sar(data, n_sar):
    dimsin = data.shape
    if np.size(dimsin) == 1:
        data_rshp = data.reshape((int(dimsin[0] / n_sar), int(n_sar)))
    else:
        data_rshp = data.reshape((int(dimsin[0] / n_sar), int(n_sar), dimsin[1]))
    # DFT in azimuth
    data_ufcs = np.fft.fft(data_rshp, axis=1)
    # focused average intensity
    int_ufcs = np.mean(np.abs(data_ufcs)**2, axis=0)
    return int_ufcs, data_ufcs


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
    wavenum_scale = f_axis * 2 * np.pi / const.c * np.sin(np.radians(cfg.radar.inc_angle))

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

    slc = []

    ########################
    # PROCESSING MAIN LOOP #
    ########################

    if plot_raw:
        plt.figure()
        plt.imshow(np.real(raw_data[0]), vmin=-np.max(np.abs(raw_data[0])), vmax=np.max(np.abs(raw_data[0])),
                   origin='lower', aspect=np.float(raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                   cmap='viridis')
        #plt.title()
        plt.xlabel("Range [samples]")
        plt.ylabel("Azimuth [samples")
        plt.savefig(plot_path + os.sep  +'plot_raw_real.%s' % (plot_format), dpi=150)
        plt.close()
        plt.figure()
        plt.imshow(np.abs(raw_data[0]), vmin=0, vmax=np.max(np.abs(raw_data[0])),
                   origin='lower', aspect=np.float(raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                   cmap='viridis')
        #plt.title()
        plt.xlabel("Range [samples]")
        plt.ylabel("Azimuth [samples")
        plt.savefig(plot_path + os.sep  +'plot_raw_abs.%s' % (plot_format), dpi=150)
        plt.close()
        # utils.image(np.imag(raw_data[0]), min=-np.max(np.abs(raw_data[0])), max=np.max(np.abs(raw_data[0])), cmap='gray',
        #             aspect=np.float(
        #                 raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
        #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
        #             usetex=plot_tex,
        #             save=plot_save, save_path=plot_path + os.sep +
        #             'plot_raw_imag.%s' % (plot_format),
        #             dpi=150)
        # utils.image(np.abs(raw_data[0]), min=0, max=np.max(np.abs(raw_data[0])), cmap='gray',
        #             aspect=np.float(
        #                 raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
        #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
        #             usetex=plot_tex,
        #             save=plot_save, save_path=plot_path + os.sep +
        #             'plot_raw_amp.%s' % (plot_format),
        #             dpi=150)
        # utils.image(np.angle(raw_data[0]), min=-np.pi, max=np.pi, cmap='gray',
        #             aspect=np.float(
        #                 raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
        #             title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
        #             usetex=plot_tex, save=plot_save,
        #             save_path=plot_path + os.sep +
        #             'plot_raw_phase.%s' % (plot_format),
        #             dpi=150)

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
    info.msg("Pulse-pair processing")
    (dop_pp_avg, dop_pha_avg, coh,
     mean_int_profile, int_spe,
     int_spe_smth) = pulse_pair(data[0:az_size_orig, 0:rg_size_orig], prf)
    info.msg("Mean DCA (pulse-pair average): %f Hz" % (np.mean(dop_pp_avg)))
    info.msg("Mean DCA (pulse-pair phase average): %f Hz" % (np.mean(dop_pha_avg)))
    info.msg("Mean coherence: %f " % (np.mean(coh)))
    info.msg("Saving output to %s" % (pp_file))
    # Unfocused SAR
    info.msg("Unfocused SAR")
    int_unfcs, data_ufcs = unfocused_sar(data[0:az_size_orig, 0:rg_size_orig], cfg.processing.n_sar)
    plt.figure()
    plt.imshow(np.fft.fftshift(int_unfcs, axes=(0,)),
               origin='lower', aspect='auto',
               cmap='viridis')
    # plt.title()
    plt.xlabel("Range [samples]")
    plt.ylabel("Doppler [samples")
    plt.savefig(plot_path + os.sep + 'ufcs_int.%s' % (plot_format), dpi=150)
    plt.close()
    plt.figure()
    plt.plot(mean_int_profile)
    plt.xlabel("Range samples [Pixels]")
    plt.ylabel("Intensity")
    plt.savefig(plot_path + os.sep + 'mean_int.%s' % (plot_format))
    plt.figure()
    plt.plot(2 * wavenum_scale[1:np.int(cfg.radar.n_rg/2)], (np.abs(int_spe[1:np.int(cfg.radar.n_rg/2)])))
    plt.plot(2 * wavenum_scale[1:np.int(cfg.radar.n_rg/2)], (np.abs(int_spe_smth[1:np.int(cfg.radar.n_rg/2)])))
    plt.xlabel("Delta_k [rad/m]")
    plt.ylabel("Power")
    plt.savefig(plot_path + os.sep + 'int_spe.%s' % (plot_format))
    np.savez(pp_file,
             dop_pp_avg=dop_pp_avg,
             dop_pha_avg=dop_pha_avg,
             coh=coh,
             ufcs_intensity=int_unfcs,
             mean_int_profile=mean_int_profile,
             int_spe=int_spe,
             int_spe_smth=int_spe_smth, wavenum_scale=wavenum_scale)

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
        RCS_power = np.zeros_like(analysis_deltan)

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
                         raw_data, RCS_power, analysis_deltan, path_p)
        # Calculating the required delta_f value
        delta_f = cal_delta_f(inc, wave_scale)
        delta_k = delta_f / const.c
        ind_N = np.int(round(delta_k * (2 * Scene_scope) * 2))
        # sar processing parameters transfor
        s_r, r_int_num, ind_N, data_rshp, s_a, az_int, analyse_num, n_sar_a = Transform(raw_data, R_samples,
                    n_sar_r, r_int_num, Az_smaples, ind_N, az_int, n_sar_a, analyse_num, rang_img, Azi_img)
        # initialization parameters
        dk_higha = np.zeros((np.size(analyse_num), r_int_num), dtype = np.float)
        RCS_power = np.zeros_like(analysis_deltan)
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


def delta_k_spectrum(Scene_scope, r_int_num, inc,
                     raw_data, RCS_power,
                     analysis_deltan, path_p):

    spck_f = np.fft.fftshift(np.fft.fft(raw_data, axis=1), axes=(1,))

    analysis_deltaf = analysis_deltan / 4 / Scene_scope * const.c

    for ind in range(np.size(analysis_deltan)):
        dk_ind = spck_f[:, 0:r_int_num] * np.conj(spck_f[:, ind:ind + r_int_num])
        RCS_power[ind] = np.abs(np.mean(dk_ind))

    xx = analysis_deltaf[1:np.size(analysis_deltan)] / 1e6
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(xx, 10*(np.log10(RCS_power[1:np.size(analysis_deltan)])))
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
    Proc = True
    if Proc:
        skim_process(args.cfg_file, args.raw_file)
        delta_k_processing(args.raw_file, args.cfg_file)
    else:
        delta_k_processing(args.raw_file, args.cfg_file)
#    delta_k_processing(r'C:\Users\lyh\Documents\SKIM_12deg_rar_0\Time0_inc_s6_azimuth_s0\raw_data.nc', r'C:\Users\lyh\Documents\SKIM_12deg_rar_0\Time0_inc_s6_azimuth_s0\config.cfg')
