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


def skim_process(cfg_file, raw_output_file, output_file):

    ###################
    # INITIALIZATIONS #
    ###################

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SKIM Processor: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('-------------------------------------------------------------------')

    # CONFIGURATION FILE
    cfg = tpio.ConfigFile(cfg_file)

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
    sr0 = raw_file.get('sr0')
    azimuth = raw_file.get('azimuth')
    raw_file.close()

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
        utils.image(np.real(raw_data[0]), min=-np.max(np.abs(raw_data[0])), max=np.max(np.abs(raw_data[0])), cmap='gray',
                    aspect=np.float(
                        raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                    title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                    usetex=plot_tex,
                    save=plot_save, save_path=plot_path + os.sep +
                    'plot_raw_real.%s' % (plot_format),
                    dpi=150)
        utils.image(np.imag(raw_data[0]), min=-np.max(np.abs(raw_data[0])), max=np.max(np.abs(raw_data[0])), cmap='gray',
                    aspect=np.float(
                        raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                    title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                    usetex=plot_tex,
                    save=plot_save, save_path=plot_path + os.sep +
                    'plot_raw_imag.%s' % (plot_format),
                    dpi=150)
        utils.image(np.abs(raw_data[0]), min=0, max=np.max(np.abs(raw_data[0])), cmap='gray',
                    aspect=np.float(
                        raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                    title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                    usetex=plot_tex,
                    save=plot_save, save_path=plot_path + os.sep +
                    'plot_raw_amp.%s' % (plot_format),
                    dpi=150)
        utils.image(np.angle(raw_data[0]), min=-np.pi, max=np.pi, cmap='gray',
                    aspect=np.float(
                        raw_data[0].shape[1]) / np.float(raw_data[0].shape[0]),
                    title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                    usetex=plot_tex, save=plot_save,
                    save_path=plot_path + os.sep +
                    'plot_raw_phase.%s' % (plot_format),
                    dpi=150)

        # Optimize matrix sizes
        az_size_orig, rg_size_orig = raw_data[0].shape
        optsize = utils.optimize_fftsize(raw_data[0].shape)
        optsize = [raw_data.shape[0], optsize[0], optsize[1]]
        data = np.zeros(optsize, dtype=complex)


    print('-----------------------------------------')
    print(time.strftime(
        "Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-----------------------------------------')


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-r', '--raw_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    skim_process(args.cfg_file, args.raw_file, args.output_file)
