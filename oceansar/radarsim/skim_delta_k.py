#!/usr/bin/env python

""" ============================================
                 Debug file
    ============================================

        Script to Delta-k processing

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



def raw_data_extraction(raw_output_file):

    ###################
    # INITIALIZATIONS #
    ###################
     # RAW DATA
    raw_file = tpio.RawFile(raw_output_file, 'r')
    raw_data = raw_file.get('raw_data*')
    #dop_ref = raw_file.get('dop_ref')
    #sr0 = raw_file.get('sr0')
    #azimuth = raw_file.get('azimuth')
    raw_file.close()
    #n_az
    return raw_data
    
    
def delta_k_processing(raw_data, cfg_file):
    #parameters
    cfg = tpio.ConfigFile(cfg_file)
    Az_smaples = cfg.radar.n_pulses
    Az_smaples = 2048
    PRF = cfg.radar.prf
    fs = cfg.radar.Fs
    R_samples = cfg.radar.n_rg
    #Sp_x = cfg.ocean.dx
    inc = cfg.radar.inc_angle
    
    #processing parameters
    analysis_deltan =  np.linspace(0,300,301) #for delta-k spectrum analysis
    RCS_power = np.zeros_like(analysis_deltan)
    wave_scale = 100#44.1#44.1#29.3#26.4#100##25.3#26.21#26.5##25.34#26.31##10# 
    r_int_num = 700
    r_int_num_fd = 400
    az_int = 500
    num_az = 1
    list = range(1,num_az)
    analyse_num = range(2,Az_smaples - az_int -1 - num_az)
    Omiga_p = np.zeros((np.size(analyse_num),np.size(list)+1), dtype=np.float)
    Omiga_p_z = np.zeros(np.size(analyse_num), dtype=np.float)
    fil_Omiga_p = np.zeros(np.size(analyse_num), dtype=np.float)
    fd = np.zeros(np.size(analyse_num), dtype=np.float)   
    pha1 = np.zeros(np.size(analyse_num), dtype=np.float)
    Scene_scope = ((R_samples-1) * const.c / fs / 2) / 2    
    dk_higha = np.zeros((np.size(analyse_num),r_int_num), dtype='complex128')
    
    #pulse_pulse processing
    #PP = False
    
    #intensity
    raw_int = raw_data * np.conj(raw_data)
    
    if plot_pattern:
        plt.figure()
        #plt.imshow(np.log10(np.abs(RCS)))
        xs = 2 * const.c / fs * np.linspace(0,R_samples-1,R_samples)
        ys = np.linspace(0,Az_smaples-1,Az_smaples) / PRF
        plt.imshow(np.abs(raw_int))
        #plt.xlim(xs)
        #plt.ylim(ys)
        #plt.imshow(ys,xs,np.abs(raw_int))
        #plt.grid(False)
        plt.xlabel("Range [m]")
        plt.ylabel("Slow time (s)")
    
    intens_spectrum = np.mean(np.fft.fft(np.abs(raw_int), axis=1), axis=0)
    
    if plot_spectrum:
        plt.figure()
        #plt.plot(np.linspace(1,np.int(Sp_x/2),np.int(Sp_x/2)-1) / 4 / Scene_scope, 10 * np.log10(np.abs(intens_spectrum[1:np.int(Sp_x/2)]) / np.abs(intens_spectrum[1:np.int(Sp_x/2)]).max()))
        plt.plot(10 * np.log10(np.abs(intens_spectrum[1:np.int(R_samples/2)])))
        plt.xlabel("Delta_k [1/m]")
        plt.ylabel("Power (dB)")
    
    spck_f = np.fft.fftshift(np.fft.fft(raw_data, axis=1), axes=(1,0))
    #Calculating the required delta_f value
    delta_f = cal_delta_f(const.c, inc, wave_scale)
    delta_k = delta_f / const.c
    ind_N = np.int(round(delta_k * (2 * Scene_scope) * 2))
        
    
    #Extracting the information of wave_scale
    dk_high = spck_f[:,0:r_int_num] * np.conj(spck_f[:,ind_N:ind_N+r_int_num])
    
    analysis_deltaf  = analysis_deltan / 4 / Scene_scope  * const.c
    
    for ind in range(np.size(analysis_deltan)):
        dk_ind = spck_f[:,0:r_int_num] * np.conj(spck_f[:,ind:ind+r_int_num])
        RCS_power[ind] = np.mean(np.abs(np.mean(dk_ind,axis=0)))
        #RCS_power[ind] = np.abs(np.mean(dk_ind))
    
    if np.size(list)>0:
        for iii in list:     
            for ind in range(np.size(analyse_num)):
                
            #dk_ind = spck_f[:,0:-1-ind] * np.conj(spck_f[:,ind:-1]) * amp_delta_k
                dk_inda = spck_f[iii:,0:r_int_num] * np.conj(spck_f[0:-iii,ind_N:ind_N+r_int_num])
                #RCS_power[ind_N] = RCS_power[ind_N] + np.mean(np.abs(np.mean(dk_ind1,axis=0)))
                #dk_higha[ind,:] = np.mean(dk_ind1, axis=0)
                    
            
                #estimate the wave velocity
                dk_higha = np.mean(dk_inda, axis=1)
                #dk_higha = np.mean(dk_high, axis=1)
                #for ind in range(np.size(analyse_num)):
                           #pha = -np.angle(np.mean(dk_higha[analyse_num[ind]:] * np.conj(dk_higha[0:-analyse_num[ind]])))
                pha = -np.angle(np.mean(dk_higha[analyse_num[ind]:analyse_num[ind]+az_int] * np.conj(dk_higha[0:az_int])))
                #pha1[ind] = pha
                Omiga_p[ind,iii] = pha / analyse_num[ind] * PRF 
            
            #dk_ind = spck_f[:,0:-1-ind] * np.conj(spck_f[:,ind:-1]) * amp_delta_k
            print(iii / (np.size(list)+1))
        dk_higha = np.mean(dk_high, axis=1)
        for ind in range(np.size(analyse_num)):
            pha = -np.angle(np.mean(dk_higha[analyse_num[ind]:analyse_num[ind]+az_int] * np.conj(dk_higha[0:az_int])))
            Omiga_p_z[ind] = pha / analyse_num[ind] * PRF 
        
        Omiga_p[:,0] = Omiga_p_z 
        fil_Omiga_p = np.mean(Omiga_p,axis=1) 
    else:
        dk_higha = np.mean(dk_high, axis=1)
        for ind in range(np.size(analyse_num)):
            pha = -np.angle(np.mean(dk_higha[analyse_num[ind]:analyse_num[ind]+az_int] * np.conj(dk_higha[0:az_int])))
            Omiga_p_z[ind] = pha / analyse_num[ind] * PRF 
        fil_Omiga_p = Omiga_p_z
        #print(v_p)
    
    #Doppler frequency estimation with different time intervals
    #for ind in range(np.size(analyse_num)):
     #   dk_ind_t = spck_f[0,0:r_int_num_fd] * np.conj(spck_f[analyse_num[ind],ind_N:ind_N+r_int_num_fd]) 
     #   fd[ind] = np.angle(np.mean(dk_ind_t)) / (analyse_num[ind] / PRF) / 2 / np.pi -  Omiga_p[ind] / 2 / np.pi
        #print(fd[inn,0])   

    #Maxim = np.max(RCS_power[1:np.size(analysis_deltan)])
    xx = analysis_deltaf[1:np.size(analysis_deltan)] / 1e6
    xxn = const.c / 2 / analysis_deltaf[1:np.size(analysis_deltan)] / np.sin(12 * np.pi /180)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(xx, 10*(np.log10(RCS_power[1:np.size(analysis_deltan)])));
    #ax2 = ax1.twiny() # this is the important function
    #ax2.plot(xx,xxn)
    #ax2.invert_xaxis()
        #plt.plot(xx, 10*(np.log10(RCS_power[1:np.size(analysis_deltan)])))
    ax1.set_xlabel("Delta_f [MHz]")
    #ax2.set_xlabel("Wave_length [m]")
    ax1.set_ylabel("Power (dB)")


    plt.figure()
    plt.plot(analyse_num, fil_Omiga_p)
    plt.xlabel("Azimuth interval [Pixel]")
    plt.ylabel("Angular velocity (rad/s)") 
    
    #Remove polyfit
    #poly = np.polyfit(analyse_num,fil_Omiga_p,deg=7)
    #z = np.polyval(poly, analyse_num)
    #plt.plot(analyse_num, z)
    #print(z)
    
    if np.size(list)>0:
        return Omiga_p
    else:
        return Omiga_p_z
    

    #plt.figure()
    #plt.plot(analyse_num, pha1)
    #plt.xlabel("Azimuth interval [Pixel]")
    #plt.ylabel("Phase variation (rad)")     
    
    #plt.figure()
    #plt.plot(analyse_num, fd)
    #plt.xlabel("Azimuth interval [Pixel]")
    #plt.ylabel("fd (Hz)")   
    
       
def cal_delta_f(light_velocity = 0, inc = 0, wave_scale = 10):
    
    delta_f = light_velocity / 2 / wave_scale / np.sin(inc * np.pi /180)
    
    return delta_f
   


if __name__ == '__main__':
    
    plot_pattern = True
    plot_spectrum = False
    PP = False

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-r', '--raw_file')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()

    #skim_process(args.cfg_file, args.raw_file, args.output_file)
    #raw_data = raw_data_extraction(raw_output_file=r'D:\research\TU Delft\Data\OceanSAR\SKIM_12deg_rar_2048_wind\raw_data.nc')
    raw_data = raw_data_extraction(raw_output_file=r'D:\research\TU Delft\Data\OceanSAR\SKIM_12deg_rar_2048_swell\raw_data.nc')
    Omiga = delta_k_processing(raw_data[0], cfg_file = r'D:\research\TU Delft\Data\OceanSAR\SKIM_proxy.cfg')
    
    if PP:
        Omiga_1 = Omiga[:,0]
    else:
        Omiga_1 = Omiga 
        
    threshold = 0.5#0.5#5
    stp = 500
    Omiga_b = Omiga_1
    for iii in range(1,stp):
       if (np.std(Omiga_1)>threshold):
           if PP:
               Omiga_b = Omiga[((iii+1)*10):,:]
               Omiga_1 = Omiga_b[:,0]
           else:
               Omiga_b = Omiga[((iii+1)*10):]
               Omiga_1 = Omiga_b              
       else:
           print((iii+1)*10)
           print(np.mean(Omiga_b))
           break
    

