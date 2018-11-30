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
from oceansar import io as tpio
from oceansar import constants as const
from oceansar.radarsim.antenna import sinc_1tx_nrx


class prinfo(object):
    def __init__(self, verbosity, header='processing'):
        self.verbosity = verbosity
        self.header = header

    def msg(self, message, importance=2):
        if importance > (2 - self.verbosity):
            print("%s -- %s" % (self.header, message))


def pulse_pair(data, prf):
    pp = data[1:] * np.conj(data[0:-1])
    # Phase to dop
    p2d = 1 / (2 * np.pi) * prf
    # average complex phasors
    pp_rg_avg = np.mean(pp, axis=-1)
    pp_rg_avg_dop = p2d * np.angle(pp_rg_avg)
    # average phase to eliminate biases due to amplitude variations
    phase_rg_avg_dop = p2d * np.mean(np.angle(pp), axis=-1)
    # Coherence
    coh_rg_avg = pp_rg_avg / np.sqrt(np.mean(np.abs(data[1:])**2, axis=-1) *
                                     np.mean(np.abs(data[0:-1])**2, axis=-1))
    return pp_rg_avg_dop, phase_rg_avg_dop, np.abs(coh_rg_avg)


def skim_process(cfg_file, raw_output_file, output_file):

    ###################
    # INITIALIZATIONS #
    ###################

    # CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)
    info = prinfo(cfg.sim.verbosity, "processor")
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
    plot_path = os.path.dirname(output_file) + os.sep + plot_path
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
                                   np.exp((2j * np.pi) * t_vec * dop_ref.reshape((1, rg_size_orig))))
    # Pulse pair
    info.msg("Pulse-pair processing")
    dop_pp_avg, dop_pha_avg, coh = pulse_pair(data[0:az_size_orig, 0:rg_size_orig], prf)
    info.msg("Mean DCA (pulse-pair average): %f Hz" % (np.mean(dop_pp_avg)))
    info.msg("Mean DCA (pulse-pair phase average): %f Hz" % (np.mean(dop_pha_avg)))
    info.msg("Mean coherence: %f " % (np.mean(coh)))
    info.msg("Saving output to %s" % (output_file))
    np.savez(output_file,
             dop_pp_avg=dop_pp_avg,
             coh=coh)

    info.msg(time.strftime("All done [%Y-%m-%d %H:%M:%S]", time.localtime()))


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-r', '--raw_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    skim_process(args.cfg_file, args.raw_file, args.output_file)
