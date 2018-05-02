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

osr_script = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'oceansar_sarsim.py'
# TODO: make this part of the configuration file
## Definition of simulation parameters to be varied, this sho
# Wind speed [m/s]
v_wind_U = [5, 7.5, 10]
# Wind direction [deg]
v_wind_dir = np.arange(18, dtype=np.float32)*20
# Surface current magnitude [m/s]
v_current_mag = [0.]
# Surface current direction [deg]
v_current_dir = [0]
 # SAR incidence angle [deg]
v_inc_angle = [35.]
n_rep = 1
cfg_file_name = 'config.cfg'

def batch_sarsim(template_file):
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
    step = 0
    sim_path_ref = ref_cfg.sim.path + os.sep + 'sim_U%d_wdir%d_smag%d_sdir%d_inc%d_i%i'
    n_all = (np.size(v_wind_U) * np.size(v_wind_dir) *
             np.size(v_current_mag) * np.size(v_current_dir) *
             np.size(v_inc_angle) * n_rep)
    for wind_U in v_wind_U:
        for wind_dir in v_wind_dir:
            for current_mag in v_current_mag:
                for current_dir in v_current_dir:
                    for inc_angle in v_inc_angle:
                        for i_rep in np.arange(n_rep, dtype=np.int):
                            step = step + 1
                            print("")
                            print('CONFIGURATION AND LAUNCH OF SIMULATION %d of %d' % (step, n_all))

                            ### CONFIGURATION FILE ###
                            # Load template
                            cfg = ref_cfg

                            # Modify template, create directory & save
                            cfg.sim.path = sim_path_ref % (wind_U, wind_dir,
                                                           current_mag,
                                                           current_dir,
                                                           inc_angle,
                                                           i_rep)
                            cfg.ocean.wind_U = wind_U
                            cfg.ocean.wind_dir = wind_dir
                            cfg.ocean.current_mag = current_mag
                            cfg.ocean.current_dir = current_dir
                            cfg.sar.inc_angle = inc_angle

                            if not os.path.exists(cfg.sim.path):
                                os.makedirs(cfg.sim.path)

                            # Save configuration file into an alternate file
                            cfg.save(cfg.sim.path + os.sep + cfg_file_name)

                            ### LAUNCH MACSAR ###
                            subprocess.call([sys.executable, osr_script, cfg.sim.path + os.sep + cfg_file_name])


def postprocess_batch_sim(template_file, plots=True):
    cfg_file = utils.get_parFile(parfile=template_file)
    ref_cfg = osrio.ConfigFile(cfg_file)
    step = 0
    npol = (2 if ref_cfg.sar.pol == 'DP' else 1)
    nch = ref_cfg.sar.num_ch
    nim = nch * npol


    sim_path_ref = ref_cfg.sim.path + os.sep + 'sim_U%d_wdir%d_smag%d_sdir%d_inc%d_i%i'
    n_all = (np.size(v_wind_U) * np.size(v_wind_dir) *
             np.size(v_current_mag) * np.size(v_current_dir) *
             np.size(v_inc_angle) * n_rep)
    mean_cohs = np.zeros((np.size(v_wind_U), np.size(v_wind_dir),
                          np.size(v_current_mag), np.size(v_current_dir),
                          np.size(v_inc_angle), n_rep, (nim) * (nim - 1) / 2), dtype=np.complex)
    mean_abscohs = np.zeros((np.size(v_wind_U), np.size(v_wind_dir),
                             np.size(v_current_mag), np.size(v_current_dir),
                             np.size(v_inc_angle), n_rep, (nim) * (nim - 1) / 2), dtype=np.complex)

    for ind_w_U in range(np.size(v_wind_U)):
        for ind_w_dir in range(np.size(v_wind_dir)):
            for ind_c_mag in range(np.size(v_current_mag)):
                for ind_c_dir in range(np.size(v_current_dir)):
                    for ind_inc in range(np.size(v_inc_angle)):
                        for i_rep in range(n_rep):
                            # Read data
                            path = sim_path_ref % (v_wind_U[ind_w_U], v_wind_dir[ind_w_dir],
                                                   v_current_mag[ind_c_mag],
                                                   v_current_dir[ind_c_dir],
                                                   v_inc_angle[ind_inc],
                                                   i_rep)
                            try:
                                data_filename = os.path.join(path, 'ati_stats.npz')
                                npzfile = np.load(data_filename)
                                mean_cohs[ind_w_U, ind_w_dir, ind_c_mag, ind_c_dir, ind_inc, i_rep] = npzfile['coh_mean']
                                mean_abscohs[ind_w_U, ind_w_dir, ind_c_mag, ind_c_dir, ind_inc, i_rep] = npzfile['abscoh_mean']
                                coh_lut= npzfile['coh_lut']
                            except OSError:
                                print("Issues with %s" % data_filename)

    return mean_cohs, mean_abscohs, coh_lut


if __name__ == '__main__':
    # INPUT ARGUMENTS

    if len(sys.argv) < 2:
        print("You need to pass a reference configuration file")
    else:
        batch_sarsim(sys.argv[1])
