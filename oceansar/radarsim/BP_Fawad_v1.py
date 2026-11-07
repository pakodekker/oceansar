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
from back_projection_func import back_projection
from oceansar import ocs_io as tpio
from oceansar import constants as const
from oceansar.radarsim.antenna import sinc_1tx_nrx, sinc_bp


def sar_focus(cfg_file, raw_output_file, output_file):

    ###################
    # INITIALIZATIONS #
    ###################

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR Processor: %Y-%m-%d %H:%M:%S", time.localtime()))
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

    # SAR
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    num_ch = cfg.sar.num_ch
    alt = cfg.sar.alt
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.sar.rg_bw
    over_fs = cfg.sar.over_fs

    # CALCULATE PARAMETERS
    l0 = const.c / f0
    if v_ground == 'auto':
        v_ground = geo.orbit_to_vel(alt, ground=True)
    rg_sampling = rg_bw * over_fs

    # RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    sr0 = raw_file.get('sr0')
    inc_angle = raw_file.get('inc_angle')
    b_ati = raw_file.get('b_ati')
    b_xti = raw_file.get('b_xti')
    raw_file.close()

    
    # OTHER INITIALIZATIONS
    # Create plots directory
    plot_path = os.path.dirname(output_file) + os.sep + plot_path
    if plot_save:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    slc = []
    BP = 1
    ########################
    # PROCESSING MAIN LOOP #
    ########################
    for ch in np.arange(num_ch):

        if plot_raw:
            plt.figure()
            plt.imshow(np.real(raw_data[0, ch]),
                       vmin=-np.max(np.abs(raw_data[0, ch])),
                       vmax=np.max(np.abs(raw_data[0, ch])), cmap='gray')
            plt.savefig(plot_path + os.sep + ('plot_raw_real_%d.%s' % (ch, plot_format)))
            plt.close()
            
        if BP ==1:
                    # Getting each channel data
            az_size_orig, rg_size_orig = raw_data[0, 0].shape
            backscatter = np.zeros((num_ch, az_size_orig, rg_size_orig)) 
            az_size_orig, rg_size_orig = raw_data[0, ch].shape  # az 142 and rg 97
            data = np.zeros((az_size_orig, rg_size_orig), dtype=complex)  #Initialization 2D, ( az, rg)
            data = raw_data[0, ch, :, :]    
            backscatter[ch, :, :] = back_projection(data, prf, f0, v_ground, sr0, rg_sampling)
            
            # Removal of non valid samples
            n_val_az_2 = np.floor(
            doppler_bw / 2. / (2. * v_ground**2. / l0 / sr0) * prf / 2.) * 2.
            reduced = backscatter[ch, int(n_val_az_2):int(az_size_orig - n_val_az_2 - 1), :]
            print(reduced.shape)
        
            idx_reduced_azi = np.sum(np.abs(reduced), axis=1)
            intensity_BP = np.abs((reduced[np.argmax(idx_reduced_azi), :]))  #Sliced at Max Azimuth
            plt.figure()
            plt.imshow(np.abs(reduced), origin='lower', vmin=0, vmax=np.max(np.abs(reduced)),
                       aspect=float(rg_size_orig) / float(az_size_orig),
                       cmap='gray')
            plt.xlabel("Range")
            plt.title("Reduced BP")
            plt.ylabel("Azimuth")
            
           
           

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
 
        fr = np.fft.fftfreq(rg_size, 1/rg_sampling)
        fa = np.fft.fftfreq(az_size, 1/prf)
        ## Compensation of ANTENNA PATTERN
        ## FIXME this will not work for a long separation betwen Tx and Rx!!!
        sin_az = fa * l0 / (2 * v_ground)
        if hasattr(cfg.sar, 'ant_L'):
            ant_L = cfg.sar.ant_L
            if cfg.sar.L_total:
                beam_pattern = sinc_1tx_nrx(sin_az, ant_L * num_ch, f0, num_ch, field=True)
            else:
                beam_pattern = sinc_1tx_nrx(sin_az, ant_L, f0, 1, field=True)
        else:
            ant_l_tx = cfg.sar.ant_L_tx
            ant_l_rx = cfg.sar.ant_L_rx
            beam_pattern = (sinc_bp(sin_az, ant_l_tx, f0, field=True)
                            * sinc_bp(sin_az, ant_l_rx, f0, field=True))


        #fa[az_size/2:] = fa[az_size/2:] - prf
        rcmc_fa = sr0 / np.sqrt(1 - (fa * (l0 / 2.) / v_ground)**2.) - sr0
        data = np.fft.fft(np.fft.fft(data, axis=-1), axis=-2)
       
#        for i in np.arange(az_size):
#            data[i,:] *= np.exp(1j*2*np.pi*2*rcmc_fa[i]/const.c*fr)
        data = (data * np.exp(4j * np.pi * rcmc_fa.reshape((1, az_size, 1)) /
                              const.c * fr.reshape((1, 1, rg_size))))
    
        data = np.fft.ifft(data, axis=2)
       
        if plot_rcmc_dopp:
            plt.figure()
            plt.imshow(np.fft.fftshift(np.abs(data[0]), axes=0), vmax=np.max(np.abs(data)), cmap='gray',
                       origin='lower')
            plt.savefig(plot_path + os.sep + ('plot_rcmc_dopp_%d.%s' % (ch, plot_format)))
            plt.close()

        if plot_rcmc_time:
            rcmc_time = np.fft.ifft(data[0], axis=0)[
                :az_size_orig, :rg_size_orig]
            rcmc_time_max = np.max(np.abs(rcmc_time))
            plt.figure()
            plt.imshow(np.real(rcmc_time), vmin=-rcmc_time_max, vmax=rcmc_time_max, cmap='gray',
                       origin='lower')
            plt.savefig(plot_path + os.sep + ('plot_rcmc_time_real_%d.%s' % (ch, plot_format)))
            plt.close()

        # Azimuth compression
        print(
            'Applying azimuth compression... [Channel %d/%d]' % (ch + 1, num_ch))

        n_samp = 2 * (int(doppler_bw / (fa[1] - fa[0])) / 2)
        weighting = (az_weighting -
                     (1. - az_weighting) * np.cos(2 * np.pi * np.linspace(0, 1., int(n_samp))))
        # Compensate amplitude loss

        L_win = np.sum(np.abs(weighting)**2) / weighting.size
        weighting /= np.sqrt(L_win)
        if fa.size > n_samp:
            zeros = np.zeros(az_size)
            zeros[0:int(n_samp)] = weighting
            weighting = np.roll(zeros, int(-n_samp / 2))
        weighting = np.where(np.abs(beam_pattern) > 0, weighting/beam_pattern, 0)
        ph_ac = 4. * np.pi / l0 * sr0 * \
            (np.sqrt(1. - (fa * l0 / 2. / v_ground)**2.) - 1.)
  
        data = data * (np.exp(1j * ph_ac) * weighting).reshape((1, az_size, 1))
  
        #=====================================
        data = np.fft.ifft(data, axis=1)
        
        print(data.shape)
        
        print('Finishing... [Channel %d/%d]' % (ch + 1, num_ch))


        # Reduce to initial dimension
        data = data[:, :int(az_size_orig), :int(rg_size_orig)]
        print(data.shape)
        # Removal of non valid samples
        n_val_az_2 = np.floor(
            doppler_bw / 2. / (2. * v_ground**2. / l0 / sr0) * prf / 2.) * 2.
        # data = raw_data[ch, n_val_az_2:(az_size_orig - n_val_az_2 - 1), :]
        data = data[:, int(n_val_az_2):int(az_size_orig - n_val_az_2 - 1), :]
        
        
        print(ch)
        print(data.shape)
        if plot_image_valid:
            plt.figure()
            plt.imshow(np.abs(data[0]), origin='lower', vmin=0, vmax=np.max(np.abs(data)),
                       aspect=float(rg_size_orig) / float(az_size_orig),
                       cmap='gray')
            plt.xlabel("Range")
            plt.ylabel("Azimuth")
            plt.savefig(os.path.join(
            plot_path, ('plot_image_valid_%d.%s' % (ch, plot_format)))) 
            
            
            
            
            
        ##### INTENSITY PLOT at MAX Azimuth Position
        range_resolution = const.c / (2 * rg_sampling)
        azimuth_resolution = v_ground/prf
        print (range_resolution)
        print (azimuth_resolution)
        
        intensity_grid = np.abs(data[0]) #for frequency based
        azimuth_intensity_sum = np.sum(intensity_grid, axis=1)
        intensity_Freqbased = (intensity_grid[np.argmax(azimuth_intensity_sum), :])
        
        plt.figure(figsize=(10, 5))
        plt.plot(np.abs(intensity_Freqbased), label='Frequency-Based Intensity')
        plt.plot(np.abs(intensity_BP), label='Back-Projection Intensity')
        plt.xlabel('Range Bin')
        plt.ylabel('Magnitude')
        plt.grid()
        plt.legend()  # Place this before plt.show()
        
          

        # IN dB and X axis in Meters
        range_bins = np.arange(intensity_BP.size)
        range_resolution = const.c / (2 * rg_sampling)
        range_distances = range_bins * range_resolution
        intensity_Freqbased_dB = 10 * np.log10(np.abs(intensity_Freqbased) + 1e-6)
        intensity_BP_dB = 10 * np.log10(np.abs(intensity_BP) + 1e-6)
                
        plt.figure(figsize=(10, 6))
        plt.plot(range_distances, intensity_Freqbased_dB, label="Freq. Based Algo.")
        plt.plot(range_distances, intensity_BP_dB, label="BP Algo.")
        plt.title("Intensity Across Range Distances at Maximum Intensity Azimuth Index")
        plt.xlabel("Range Distance (meters)")
        plt.ylabel("Intensity (dB)")
        plt.legend()
        plt.grid()
        plt.show()
        
        ####
      
        
        
        slc.append(data)
    
    

    # Save processed data
    slc = np.array(slc, dtype=complex)
    print("Shape of SLC: " + str(slc.shape), flush=True)
   
    proc_file = tpio.ProcFile(output_file, 'w', slc.shape)
    proc_file.set('slc*', slc)
    proc_file.set('inc_angle', inc_angle)
    proc_file.set('f0', f0)
    proc_file.set('num_ch', num_ch)
    proc_file.set('ant_L', ant_l_tx)
    proc_file.set('prf', prf)
    proc_file.set('v_ground', v_ground)
    proc_file.set('orbit_alt', alt)
    proc_file.set('sr0', sr0)
    proc_file.set('rg_sampling', rg_bw*over_fs)
    proc_file.set('rg_bw', rg_bw)
    proc_file.set('b_ati', b_ati)
    proc_file.set('b_xti', b_xti)
    proc_file.close()

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

    sar_focus(args.cfg_file, args.raw_file, args.output_file)
