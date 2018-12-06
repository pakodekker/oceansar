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
Time = range(0,1) 
# incidence angle [deg]
inc_s = [12, 6]
# look angle [deg]
azimuth_s = [90, 60]

n_rep = 1
cfg_file_name = 'config.cfg'

def batch_skimsim(template_file):
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
    step = 0
    sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d'
    n_all = (np.size(Time) * np.size(inc_s) *
             np.size(azimuth_s) * n_rep)
    for t in Time:
        for inc_angle in inc_s:
            for azimuth in azimuth_s:
                step = step + 1
                print("")
                print('CONFIGURATION AND LAUNCH OF SIMULATION %d of %d' % (step, n_all))

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

                ### LAUNCH MACSAR ###
                subprocess.call([sys.executable, osr_script, '-c', cfg.sim.path + os.sep + cfg_file_name])


def postprocess_batch_sim(template_file, plots=True):
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
   
    sim_path_ref = ref_cfg.sim.path + os.sep + 'Time%d_inc_s%d_azimuth_s%d'
    n_all = (np.size(Time) * np.size(inc_s) *
             np.size(azimuth_s) * n_rep)
    
    path = sim_path_ref % (Time[0], inc_s[0], azimuth_s[0])
    data = np.load(os.path.join(path, 'delta_k_spectrum_plots\Angular_velocity.npy'))
    Angular_velocity_curves = np.zeros((np.size(Time), np.size(inc_s),
                          np.size(azimuth_s), int(data[2])), dtype=np.complex)
    Angular_velocity_av = np.zeros((np.size(Time), np.size(inc_s),
                          np.size(azimuth_s), 0), dtype=np.complex)
    analyse_num = data[3]
    
    for ind_t in range(np.size(Time)):
        for ind_inc in range(np.size(inc_s)):
            for ind_azimuth in range(np.size(azimuth_s)):
               
                # Read data
                path = sim_path_ref % (Time[ind_t], inc_s[ind_inc],
                                       azimuth_s[ind_azimuth])
                data = np.load(os.path.join(path, 'delta_k_spectrum_plots\Angular_velocity.npy'))
                Angular_velocity_curves[ind_t, ind_inc, ind_azimuth, :] = data[0]
                Angular_velocity_av[ind_t, ind_inc, ind_azimuth, :] = data[1]
     

    plot_path = sim_path_ref + os.sep + 'Averaging_through_time'
    for ind_inc in range(np.size(inc_s)): 
         for ind_azimuth in range(np.size(azimuth_s)):      
             value = np.mean(Angular_velocity_curves[:, ind_inc, ind_azimuth, :], axis=0)  
             av_value = np.mean( Angular_velocity_av[ind_t, ind_inc, ind_azimuth, :],axis=0)  
             
             print(inc_s[ind_inc])
             print(azimuth_s[ind_azimuth])
             print(av_value)
                 
                            
             if plots:
                 plt.figure()
                 plt.plot(analyse_num, value)
                 plt.xlabel("Azimuth interval [Pixel]")
                 plt.ylabel("Angular velocity (rad/s)") 
                 plt.title("Incidence angle = %d deg, Azimuth angle = %d deg" % (inc_s[ind_inc], azimuth_s[ind_azimuth]))
                 
                 plt.savefig(os.path.join(ref_cfg.sim.path, 'Time_averaging_angular_velocity_%d_%d.png' % (inc_s[ind_inc], azimuth_s[ind_azimuth])))                
                 plt.close()


if __name__ == '__main__':
    # INPUT ARGUMENTS

    if len(sys.argv) < 2:
        print("You need to pass a reference configuration file")
    else:
        batch_skimsim(sys.argv[1])

