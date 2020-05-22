#!/usr/bin/python
# coding=utf-8

""" =============================================
    OCEANSAR SAR Simulation script
    =============================================


    **Arguments**
        - Configuration file

e.g. python oceansar_sarsim.py d:\data\configfile\20160708_NOS.cfg

"""

import sys
import os
import time
import subprocess
from oceansar import ocs_io as osrio
from oceansar import utils
from oceansar.radarsim.sar_raw_nompi import sar_raw
from oceansar.radarsim.sar_processor import sar_focus
from oceansar.radarsim.ati_processor import ati_process
from oceansar.radarsim.insar_processor import insar_process

def sarsim(cfg_file=None):

    print('-------------------------------------------------------------------', flush=True)
    print(time.strftime("OCEANSAR SAR Simulator [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
    # print('Copyright (c) Gerard Marull Paretas, Paco López-Dekker')
    print('-------------------------------------------------------------------', flush=True)
    cfg_file = utils.get_parFile(parfile=cfg_file)
    cfg = osrio.ConfigFile(cfg_file)
    # Create output directory if it doesnt exist already
    os.makedirs(cfg.sim.path, exist_ok=True)
    src_path = os.path.dirname(os.path.abspath(__file__))

    # RAW
    if cfg.sim.raw_run:

        print('Launching SAR RAW Generator...')
        sar_raw(cfg.cfg_file_name,
                os.path.join(cfg.sim.path, cfg.sim.raw_file),
                os.path.join(cfg.sim.path, cfg.sim.ocean_file),
                cfg.sim.ocean_reuse,
                os.path.join(cfg.sim.path, cfg.sim.errors_file),
                cfg.sim.errors_reuse, plot_save=True)

    # Processing
    if cfg.sim.proc_run:
        print('Launching SAR RAW Processor...')
        sar_focus(cfg.cfg_file_name, os.path.join(cfg.sim.path, cfg.sim.raw_file),
                  os.path.join(cfg.sim.path, cfg.sim.proc_file))

    if cfg.sim.insar_run:
        print('Launching InSAR L1b Processor...')
        insar_process(cfg.cfg_file_name,
                      os.path.join(cfg.sim.path, cfg.sim.proc_file),
                      os.path.join(cfg.sim.path, cfg.sim.ocean_file),
                      os.path.join(cfg.sim.path, cfg.sim.insar_file))

    if cfg.sim.ati_run:
        print('Launching SAR ATI Processor...')
        ati_process(cfg.cfg_file_name,
                    os.path.join(cfg.sim.path, cfg.sim.proc_file),
                    os.path.join(cfg.sim.path, cfg.sim.ocean_file),
                    os.path.join(cfg.sim.path, cfg.sim.ati_file))

    if cfg.sim.L2_wavespectrum_run:
        print('Launching L2 Wavespectrum Processor...')



    print('----------------------------------', flush=True)
    print(time.strftime("End of tasks [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
    print('----------------------------------', flush=True)


if __name__ == '__main__':

    # INPUT ARGUMENTS
    if len(sys.argv) < 2:
        sarsim()
    else:
        sarsim(sys.argv[1])
