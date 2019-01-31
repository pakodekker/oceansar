#!/usr/bin/python
# coding=utf-8

""" =============================================
    OCEANSAR SAR Simulation script
    =============================================


    **Arguments**
        - Configuration file

e.g. python oceansar_batchsarsim.py d:\data\configfile\20160708_NOS.cfg

"""

import sys
import os
import time
import subprocess
import numpy as np
from oceansar import io as osrio
from oceansar import utils
from matplotlib import pyplot as plt
import matplotlib as mpl

osr_script = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'oceansar_skimsim.py'
# TODO: make this part of the configuration file
## Definition of simulation parameters to be varied, this sho
<<<<<<< HEAD
<<<<<<< HEAD
Time = range(0,15) 
# incidence angle [deg]
inc_s = [12, 6]
# look angle [deg]
azimuth_s = [0, 45, 90, 135, 180, 225, 270, 315]
No = np.arange(0,20)

=======
=======
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
Time = range(0,5) 
# incidence angle [deg]
inc_s = [12, 6]
# look angle [deg]
azimuth_s = [90, 60, 30 ,0]

wave_scale = [100, 37.5, 25, 12.5]
#wave_scale = [100, 37.5, 25, 12.5]
<<<<<<< HEAD
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
=======
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py

n_rep = 1
cfg_file_name = 'config.cfg'

def batch_skimsim(template_file):
<<<<<<< HEAD
<<<<<<< HEAD
    
    for iii in range(np.size(No)):
        cfg_file = utils.get_parFile(parfile=template_file)
        ref_cfg = osrio.ConfigFile(cfg_file)
        path_m = ref_cfg.sim.path + os.sep + 'SKIM_12deg_rar_%d'%(No[iii])
        sim_path_ref = path_m + os.sep + 'Time%d_inc_s%d_azimuth_s%d'
               
        for t in Time:
            for inc_angle in inc_s:
                for azimuth in azimuth_s:
                   
                    ### CONFIGURATION FILE ###
                    # Load template
                    cfg = ref_cfg
        
                    # Modify template, create directory & save
                    cfg.sim.path = sim_path_ref % (t, inc_angle, azimuth)
                    cfg.batch_Loop.t = t
                    cfg.radar.inc_angle = inc_angle
                    cfg.radar.azimuth = azimuth
                    
                    if not os.path.exists(cfg.sim.path):
                        os.makedirs(cfg.sim.path)
        
                    # Save configuration file into an alternate file
                    cfg.save(cfg.sim.path + os.sep + cfg_file_name)
        
=======
=======
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
    step = 0
    sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d_wavelength%1f'
    n_all = (np.size(Time) * np.size(inc_s) *
             np.size(azimuth_s) * np.size(wave_scale)*n_rep)
    for t in Time:
        for inc_angle in inc_s:
            for azimuth in azimuth_s:
                for wave_length in wave_scale:
                    step = step + 1
                    print("")
                    print('CONFIGURATION AND LAUNCH OF SIMULATION %d of %d' % (step, n_all))
    
                    ### CONFIGURATION FILE ###
                    # Load template
                    cfg = ref_cfg
    
                    # Modify template, create directory & save
                    cfg.sim.path = sim_path_ref % (t, inc_angle, azimuth, wave_length)
                    cfg.batch_Loop.t = t
                    cfg.radar.inc_angle = inc_angle
                    cfg.radar.azimuth = azimuth
                    cfg.processing.wave_scale = wave_length
                                
    
                    if not os.path.exists(cfg.sim.path):
                        os.makedirs(cfg.sim.path)
    
                    # Save configuration file into an alternate file
                    cfg.save(cfg.sim.path + os.sep + cfg_file_name)
    
<<<<<<< HEAD
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
=======
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
                    ### LAUNCH MACSAR ###
                    subprocess.call([sys.executable, osr_script, '-c', cfg.sim.path + os.sep + cfg_file_name])


def postprocess_batch_sim(template_file, plots=True):
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
   
<<<<<<< HEAD
<<<<<<< HEAD
    if cfg.processing.Azi_img:
        sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d' + os.sep + 'wavelength%.1f_unfocus'
    else:
        sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d' + os.sep + 'wavelength%.1f'
        
    wave_scale = ref_cfg.processing.wave_scale
=======
    sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d_wavelength%1f'
    n_all = (np.size(Time) * np.size(inc_s) *
             np.size(azimuth_s) * np.size(wave_scale) * n_rep)
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
=======
    sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d_wavelength%1f'
    n_all = (np.size(Time) * np.size(inc_s) *
             np.size(azimuth_s) * np.size(wave_scale) * n_rep)
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
    
    path = sim_path_ref % (Time[0], inc_s[0], azimuth_s[0], wave_scale[0]) + os.sep + 'delta_k_spectrum_plots'
    # Angular velocity
    data = np.load(os.path.join(path, 'Angular_velocity.npy'))    
    Angular_velocity_curves = np.zeros((np.size(Time), np.size(inc_s),
                          np.size(azimuth_s), np.size(wave_scale), 
                          int(data[2])), dtype=np.complex)
    Angular_velocity_av = np.zeros((np.size(Time), np.size(inc_s),
                          np.size(azimuth_s), np.size(wave_scale), 0), dtype=np.complex)
    analyse_num = data[3]
    
    # Phase velocity
    data_p = np.load(os.path.join(path, 'Phase_velocity.npy'))    
    Phase_velocity_curves = np.zeros((np.size(Time), np.size(inc_s),
                          np.size(azimuth_s), np.size(wave_scale), 
                          int(data_p[2])), dtype=np.complex)
    Phase_velocity_av = np.zeros((np.size(Time), np.size(inc_s),
                          np.size(azimuth_s), np.size(wave_scale), 0), dtype=np.complex)
    
    for ind_t in range(np.size(Time)):
        for ind_inc in range(np.size(inc_s)):
            for ind_azimuth in range(np.size(azimuth_s)):
                for ind_wavelength in range(np.size(wave_scale)):
               
                    # Read data
                    path = sim_path_ref % (Time[ind_t], inc_s[ind_inc], azimuth_s[ind_azimuth], wave_scale[ind_wavelength]) + os.sep + 'delta_k_spectrum_plots'                
                    
                    data = np.load(os.path.join(path, 'Angular_velocity.npy'))
                    Angular_velocity_curves[ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data[0]
                    Angular_velocity_av[ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data[1]
                    
                    data_p = np.load(os.path.join(path, 'Phase_velocity.npy'))
                    Phase_velocity_curves[ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data_p[0]
                    Phase_velocity_av[ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data_p[1]
     

    plot_path = sim_path_ref + os.sep + 'Averaging_through_time'
    for ind_inc in range(np.size(inc_s)): 
         for ind_azimuth in range(np.size(azimuth_s)):  
              for ind_wavelength in range(np.size(wave_scale)):
                     value = np.mean(Angular_velocity_curves[:, ind_inc, ind_azimuth, ind_wavelength, :], axis=0)  
                     av_value = np.mean( Angular_velocity_av[ind_t, ind_inc, ind_azimuth, ind_wavelength, :],axis=0)  
                     
                     value_p = np.mean(Phase_velocity_curves[:, ind_inc, ind_azimuth, ind_wavelength, :], axis=0)  
                     av_value_p = np.mean( Phase_velocity_av[ind_t, ind_inc, ind_azimuth, ind_wavelength, :],axis=0)  
                 
                     print(inc_s[ind_inc],'deg')
                     print(azimuth_s[ind_azimuth],'deg')
                     print(wave_scale[ind_wavelength],'m')
                     print(av_value,'rad/s')
                     print(av_value,'m/s')
                         
                                    
                     if plots:
                         plt.figure()
                         plt.plot(analyse_num, value)
                         plt.xlabel("Azimuth interval [Pixel]")
                         plt.ylabel("Angular velocity (rad/s)") 
                         plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, wave_length = %.1f m" % (inc_s[ind_inc], azimuth_s[ind_azimuth], wave_scale[ind_wavelength]))
                         
                         plt.savefig(os.path.join(ref_cfg.sim.path, 'Time_averaging_angular_velocity_%d_%d_%.1f.png' % (inc_s[ind_inc], azimuth_s[ind_azimuth], wave_scale[ind_wavelength])))                
                         plt.close()
                         
                         plt.figure()
                         plt.plot(analyse_num, value_p)
                         plt.xlabel("Azimuth interval [Pixel]")
                         plt.ylabel("Phase velocity (m/s)") 
                         plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, wave_length = %.1f m" % (inc_s[ind_inc], azimuth_s[ind_azimuth], wave_scale[ind_wavelength]))
                         
                         plt.savefig(os.path.join(ref_cfg.sim.path, 'Time_averaging_angular_velocity_%d_%d_%.1f.png' % (inc_s[ind_inc], azimuth_s[ind_azimuth], wave_scale[ind_wavelength])))                
                         plt.close()
<<<<<<< HEAD
<<<<<<< HEAD
                         
def postprocess_batch_read(template_file, plots=True):
       
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
    wave_scale = ref_cfg.processing.wave_scale
    path_m = ref_cfg.sim.path + os.sep + 'SKIM_12deg_rar_%d'%(No[0])
    if ref_cfg.processing.Azi_img:
        sim_path_ref = path_m + os.sep + 'Time%d_inc_s%d_azimuth_s%d' + os.sep + 'wavelength%.1f_unfocus'
    else:
        sim_path_ref = path_m + os.sep + 'Time%d_inc_s%d_azimuth_s%d' + os.sep + 'wavelength%.1f'
    
    path = sim_path_ref % (Time[0], inc_s[0], azimuth_s[0], wave_scale[0]) + os.sep + 'delta_k_spectrum_plots'
    data = np.load(os.path.join(path, 'Angular_velocity.npy')) 
    
    analyse_num = data[3]
    si = data[2]
    
    Angular_velocity_curves = np.zeros((np.size(No)*np.size(Time), np.size(inc_s),
                              np.size(azimuth_s), np.size(wave_scale), 
                              int(data[2])), dtype=np.complex)
    Angular_velocity_av = np.zeros((np.size(No)*np.size(Time), np.size(inc_s),
                              np.size(azimuth_s), np.size(wave_scale), 0), dtype=np.complex)
    Phase_velocity_curves = np.zeros((np.size(No)*np.size(Time), np.size(inc_s),
                              np.size(azimuth_s), np.size(wave_scale), 
                              int(data[2])), dtype=np.complex)
    Phase_velocity_av = np.zeros((np.size(No)*np.size(Time), np.size(inc_s),
                              np.size(azimuth_s), np.size(wave_scale), 0), dtype=np.complex)
    
   
    for iii in range(np.size(No)):
        path_m = ref_cfg.sim.path + os.sep + 'SKIM_12deg_rar_%d'%(No[iii])
        if ref_cfg.processing.Azi_img:
            sim_path_ref = path_m + os.sep + 'Time%d_inc_s%d_azimuth_s%d' + os.sep + 'wavelength%.1f_unfocus'
        else:
            sim_path_ref = path_m + os.sep + 'Time%d_inc_s%d_azimuth_s%d' + os.sep + 'wavelength%.1f'
    
        for ind_t in range(np.size(Time)):
            for ind_inc in range(np.size(inc_s)):
                for ind_azimuth in range(np.size(azimuth_s)):
                    for ind_wavelength in range(np.size(wave_scale)):
                   
                        # Read data
                        path = sim_path_ref % (Time[ind_t], inc_s[ind_inc], azimuth_s[ind_azimuth], wave_scale[ind_wavelength]) + os.sep + 'delta_k_spectrum_plots'                
                        
                        data = np.load(os.path.join(path, 'Angular_velocity.npy'))
                        Angular_velocity_curves[iii*np.size(Time) + ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data[0]
                        Angular_velocity_av[iii*np.size(Time)+ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data[1]
                        
                        data_p = np.load(os.path.join(path, 'Phase_velocity.npy'))
                        Phase_velocity_curves[iii*np.size(Time)+ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data_p[0]
                        Phase_velocity_av[iii*np.size(Time)+ind_t, ind_inc, ind_azimuth, ind_wavelength, :] = data_p[1]
     
    if ref_cfg.processing.Azi_img:
        np.save(os.path.join(ref_cfg.sim.path, 'Angular_velocity_data_unfocus.npy'),
            [Angular_velocity_curves, Angular_velocity_av, si, analyse_num])
    
        np.save(os.path.join(ref_cfg.sim.path, 'Phase_velocity_data_unfocus.npy'),
            [Phase_velocity_curves,  Phase_velocity_av, si, analyse_num])
    else:
        np.save(os.path.join(ref_cfg.sim.path, 'Angular_velocity_data.npy'),
            [Angular_velocity_curves, Angular_velocity_av, si, analyse_num])
    
        np.save(os.path.join(ref_cfg.sim.path, 'Phase_velocity_data.npy'),
            [Phase_velocity_curves,  Phase_velocity_av, si, analyse_num])


def postprocess_batch_read_fd(template_file, plots=True):
       
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
    
    num = ref_cfg.radar.n_pulses - 1
    
    
    Doppler_av = np.zeros((np.size(No)*np.size(Time), np.size(inc_s),
                              np.size(azimuth_s), num), dtype=np.complex)
    
    coh_av = np.zeros((np.size(No)*np.size(Time), np.size(inc_s),
                              np.size(azimuth_s), num), dtype=np.complex)
       
   
    for iii in range(np.size(No)):
        path_m = ref_cfg.sim.path + os.sep + 'SKIM_12deg_rar_%d'%(No[iii])
        sim_path_ref = path_m + os.sep + 'Time%d_inc_s%d_azimuth_s%d'
    
        for ind_t in range(np.size(Time)):
            for ind_inc in range(np.size(inc_s)):
                for ind_azimuth in range(np.size(azimuth_s)):
                                       
                    # Read data
                    path = sim_path_ref % (Time[ind_t], inc_s[ind_inc], azimuth_s[ind_azimuth])                 
                    
                    data = np.load(os.path.join(path, 'pp_data.nc.npz'))
                    Doppler_av[iii*np.size(Time) + ind_t, ind_inc, ind_azimuth, :] = data['dop_pp_avg']
                    coh_av[iii*np.size(Time)+ind_t, ind_inc, ind_azimuth, :] = data['coh']
    if ref_cfg.processing.Azi_img:
        np.save(os.path.join(ref_cfg.sim.path, 'Doppler_coherence_data_unfocus.npy'),
            [Doppler_av, coh_av, num])    
    else:
        np.save(os.path.join(ref_cfg.sim.path, 'Doppler_coherence_data.npy'),
            [Doppler_av, coh_av, num])
    
    
if __name__ == '__main__':
    # INPUT ARGUMENTS
    Data_re_process = False
    #post_process = True
    process_data = True
    
    if len(sys.argv) < 2:
        print("You need to pass a reference configuration file")
    else:
        if Data_re_process:
            batch_skimsim(sys.argv[1])
        #elif post_process:
            #postprocess_batch_sim(sys.argv[1], plots=True)
        elif process_data:
            postprocess_batch_read(sys.argv[1], plots=True)
            postprocess_batch_read_fd(sys.argv[1], plots=True)
            
=======
=======
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py


if __name__ == '__main__':
    # INPUT ARGUMENTS

    if len(sys.argv) < 2:
        print("You need to pass a reference configuration file")
    else:
        batch_skimsim(sys.argv[1])
        postprocess_batch_sim(sys.argv[1], plots=True)
<<<<<<< HEAD
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py
=======
>>>>>>> parent of 60eb239... Delete oceansar_batchsarsim_skim.py

