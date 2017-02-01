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




def sar_focus(cfg_file, raw_output_file, output_file):

    ###################
    # INITIALIZATIONS #
    ###################

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR Processor: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('-------------------------------------------------------------------')

    ## CONFIGURATION FILE
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

    # SAR
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    num_ch = cfg.sar.num_ch
    alt = cfg.sar.alt
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.sar.rg_bw
    over_fs = cfg.sar.over_fs

    ## CALCULATE PARAMETERS
    l0 = const.c/f0
    if v_ground == 'auto': v_ground = geo.orbit_to_vel(alt, ground=True)
    rg_sampling = rg_bw*over_fs

    ## RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    sr0 = raw_file.get('sr0')
    raw_file.close()

    ## OTHER INITIALIZATIONS
    # Create plots directory
    plot_path = os.path.dirname(output_file) + os.sep + plot_path
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    slc = []

    ########################
    # PROCESSING MAIN LOOP #
    ########################
    for ch in np.arange(num_ch):

        if plot_raw:
            utils.image(np.real(raw_data[0, ch]), min=-np.max(np.abs(raw_data[0, ch])), max=np.max(np.abs(raw_data[0, ch])), cmap='gray',
                        aspect=np.float(raw_data[0, ch].shape[1])/np.float(raw_data[0, ch].shape[0]),
                        title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                        usetex=plot_tex,
                        save=plot_save, save_path=plot_path + os.sep + 'plot_raw_real_%d.%s' % (ch, plot_format),
                        dpi=150)
            utils.image(np.imag(raw_data[0, ch]), min=-np.max(np.abs(raw_data[0, ch])), max=np.max(np.abs(raw_data[0, ch])), cmap='gray',
                        aspect=np.float(raw_data[0, ch].shape[1])/np.float(raw_data[0, ch].shape[0]),
                        title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                        usetex=plot_tex,
                        save=plot_save, save_path=plot_path + os.sep + 'plot_raw_imag_%d.%s' % (ch, plot_format),
                        dpi=150)
            utils.image(np.abs(raw_data[0, ch]), min=0, max=np.max(np.abs(raw_data[0, ch])), cmap='gray',
                        aspect=np.float(raw_data[0, ch].shape[1])/np.float(raw_data[0, ch].shape[0]),
                        title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                        usetex=plot_tex,
                        save=plot_save, save_path=plot_path + os.sep + 'plot_raw_amp_%d.%s' % (ch, plot_format),
                        dpi=150)
            utils.image(np.angle(raw_data[0, ch]), min=-np.pi, max=np.pi, cmap='gray',
                        aspect=np.float(raw_data[0, ch].shape[1])/np.float(raw_data[0, ch].shape[0]),
                        title='Raw Data', xlabel='Range samples', ylabel='Azimuth samples',
                        usetex=plot_tex, save=plot_save,
                        save_path=plot_path + os.sep + 'plot_raw_phase_%d.%s' % (ch, plot_format),
                        dpi=150)

        # Optimize matrix sizes
        az_size_orig, rg_size_orig = raw_data[0, ch].shape
        optsize = utils.optimize_fftsize(raw_data[0, ch].shape)
        optsize = [raw_data.shape[0], optsize[0], optsize[1]]
        data = np.zeros(optsize, dtype=complex)
        data[:, :raw_data[0, ch].shape[0],
             :raw_data[0, ch].shape[1]] = raw_data[:, ch, :, :]

        az_size, rg_size = data.shape[1:]


        # RCMC Correction
        print('Applying RCMC correction... [Channel %d/%d]' % (ch + 1, num_ch))

        #fr = np.linspace(-rg_sampling/2., rg_sampling/2., rg_size)
        fr = (np.arange(rg_size) -rg_size/2) * rg_sampling/rg_size
        fr = np.roll(fr, int(-rg_size/2))

        fa = (np.arange(az_size) -az_size/2) * prf/az_size
        fa = np.roll(fa, int(-az_size/2))

        #fa[az_size/2:] = fa[az_size/2:] - prf
        rcmc_fa = sr0/np.sqrt(1 - (fa*(l0/2.)/v_ground)**2.) - sr0

        data = np.fft.fft2(data)

#        for i in np.arange(az_size):
#            data[i,:] *= np.exp(1j*2*np.pi*2*rcmc_fa[i]/const.c*fr)
        data = (data * np.exp(4j * np.pi * rcmc_fa.reshape((1, az_size, 1)) /
                              const.c * fr.reshape((1, 1, rg_size))))
        data = np.fft.ifft(data, axis=2)

        if plot_rcmc_dopp:
            utils.image(np.abs(data[0]), min=0., max=3.*np.mean(np.abs(data)), cmap='gray',
                        aspect=np.float(rg_size)/np.float(az_size), title='RCMC Data (Range Dopler Domain)',
                        usetex=plot_tex, save=plot_save, save_path=plot_path + os.sep + 'plot_rcmc_dopp_%d.%s' % (ch, plot_format))

        if plot_rcmc_time:
            rcmc_time = np.fft.ifft(data[0], axis=0)[:az_size_orig, :rg_size_orig]
            rcmc_time_max = np.max(np.abs(rcmc_time))
            utils.image(np.real(rcmc_time), min=-rcmc_time_max, max=rcmc_time_max, cmap='gray',
                        aspect=np.float(rg_size)/np.float(az_size), title='RCMC Data (Time Domain)',
                        usetex=plot_tex, save=plot_save, save_path=plot_path + os.sep + 'plot_rcmc_time_real_%d.%s' % (ch, plot_format))
            utils.image(np.imag(rcmc_time), min=-rcmc_time_max, max=rcmc_time_max, cmap='gray',
                        aspect=np.float(rg_size)/np.float(az_size), title='RCMC Data (Time Domain)',
                        usetex=plot_tex, save=plot_save, save_path=plot_path + os.sep + 'plot_rcmc_time_imag_%d.%s' % (ch, plot_format))

        # Azimuth compression
        print('Applying azimuth compression... [Channel %d/%d]' % (ch + 1, num_ch))

        n_samp = 2 * (np.int(doppler_bw/(fa[1] - fa[0]))/2)
        weighting = az_weighting - (1. - az_weighting)*np.cos(2*np.pi*np.linspace(0, 1., n_samp))
        # Compensate amplitude loss

        L_win = np.sum(np.abs(weighting)**2)/weighting.size
        weighting /= np.sqrt(L_win)
        if fa.size > n_samp:
            zeros = np.zeros(az_size)
            zeros[0:n_samp] = weighting
            #zeros[:n_samp/2] = weighting[:n_samp/2]
            #zeros[-n_samp/2:] = weighting[-n_samp/2:]
            weighting = np.roll(zeros,int(-n_samp/2))

        ph_ac = 4.*np.pi/l0*sr0*(np.sqrt(1. - (fa*l0/2./v_ground)**2.) - 1.)
#        for i in np.arange(rg_size):
#            data[:,i] *= np.exp(1j*ph_ac)*weighting
        data = data * (np.exp(1j * ph_ac) * weighting).reshape((1, az_size, 1))

        data = np.fft.ifft(data, axis=1)

        print('Finishing... [Channel %d/%d]' % (ch + 1, num_ch))
        # Reduce to initial dimension
        data = data[:, :az_size_orig, :rg_size_orig]

        # Removal of non valid samples
        n_val_az_2 = np.floor(doppler_bw/2./(2.*v_ground**2./l0/sr0)*prf/2.)*2.
        # data = raw_data[ch, n_val_az_2:(az_size_orig - n_val_az_2 - 1), :]
        data = data[:, n_val_az_2:(az_size_orig - n_val_az_2 - 1), :]
        if plot_image_valid:
            plt.figure()
            plt.imshow(np.abs(data[0]), origin='lower', vmin=0, vmax=np.max(np.abs(data)),
                       aspect=np.float(rg_size_orig)/np.float(az_size_orig),
                       cmap='gray')
            plt.xlabel("Range")
            plt.ylabel("Azimuth")
            plt.savefig(os.path.join(plot_path, ('plot_image_valid_%d.%s' % (ch, plot_format))))


        slc.append(data)


    # Save processed data
    slc = np.array(slc, dtype=np.complex)
    print("Shape of SLC: " + str(slc.shape), flush=True)
    proc_file = tpio.ProcFile(output_file, 'w', slc.shape)
    proc_file.set('slc*', slc)
    proc_file.close()


    print('-----------------------------------------')
    print(time.strftime("Processing finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('-----------------------------------------')

if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-r', '--raw_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    sar_focus(args.cfg_file, args.raw_file, args.output_file)