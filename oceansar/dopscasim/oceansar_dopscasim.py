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
from drama.io import cfg as drcfg
from oceansar import ocs_io as osrio
from oceansar import utils
from oceansar.dopscasim.dopsca_raw import dopsca_raw
from oceansar.radarsim.sar_processor import sar_focus
from oceansar.radarsim.ati_processor import ati_process
from oceansar.radarsim.insar_processor import insar_process
from oceansar.radarsim.L2_wavespectrum import l2_wavespectrum


# def dopscasim(cfg_file=None):

#     print('-------------------------------------------------------------------', flush=True)
#     print(time.strftime("OCEANSAR DOPSCA Simulator [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
#     print('-------------------------------------------------------------------', flush=True)
#     cfg_file = utils.get_parFile(parfile=cfg_file)
#     cfg = osrio.ConfigFile(cfg_file)
#     # Create output directory if it doesnt exist already
#     os.makedirs(cfg.sim.path, exist_ok=True)
#     src_path = os.path.dirname(os.path.abspath(__file__))

#     # RAW
#     if cfg.sim.raw_run:

#         print('Launching SAR RAW Generator...')

#         args = [cfg.sim.mpi_exec,
#                 '-np', str(cfg.sim.mpi_num_proc),
#                 sys.executable, src_path + os.sep + 'skim_raw.py',
#                 '-c', cfg.cfg_file_name,
#                 '-o', cfg.sim.path + os.sep + cfg.sim.raw_file,
#                 '-oc', cfg.sim.path + os.sep + cfg.sim.ocean_file,
#                 '-er', cfg.sim.path + os.sep + cfg.sim.errors_file]

#         if cfg.sim.ocean_reuse:
#             args.append('-ro')
#         if cfg.sim.errors_reuse:
#             args.append('-re')

#         returncode = subprocess.call(args)

#         if returncode != 0:
#             raise Exception('Something went wrong with SAR RAW Generator (return code %d)...' % returncode)

#     # Processing
#     if cfg.sim.proc_run:
#         print('Launching SAR RAW Processor...')

#         returncode = subprocess.call([sys.executable,
#                                       src_path + os.sep + 'skim_processor.py',
#                                       '-c', cfg.cfg_file_name,
#                                       '-r', cfg.sim.path + os.sep + cfg.sim.raw_file])
#         #,
#         #                              '-o', cfg.sim.path + os.sep + cfg.sim.pp_file])

#         if returncode != 0:
#             raise Exception('Something went wrong with SAR RAW Processor (return code %d)...' % returncode)



#     print('----------------------------------', flush=True)
#     print(time.strftime("End of tasks [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
#     print('----------------------------------', flush=True)


def dopscasim(cfg_file=None):

    print('-------------------------------------------------------------------', flush=True)
    print(time.strftime("OCEANSAR SAR Simulator [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
    # print('Copyright (c) Gerard Marull Paretas, Paco López-Dekker')
    print('-------------------------------------------------------------------', flush=True)
    cfg_file = utils.get_parFile(parfile=cfg_file)
    cfg = drcfg.ConfigFile(cfg_file)
    # Create output directory if it doesnt exist already
    os.makedirs(cfg.sim.path, exist_ok=True)
    src_path = os.path.dirname(os.path.abspath(__file__))

    # RAW
    if cfg.sim.raw_run:

        print('Launching SAR RAW Generator...')
        dopsca_raw(cfg.cfg_file_name,
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
                    os.path.join(cfg.sim.path, cfg.sim.insar_file),
                    os.path.join(cfg.sim.path, cfg.sim.ocean_file),
                    os.path.join(cfg.sim.path, cfg.sim.ati_file))

    if cfg.sim.L2_wavespectrum_run:
        print('Launching L2 Wavespectrum Processor...')
        l2_wavespectrum(cfg.cfg_file_name,
                        os.path.join(cfg.sim.path, cfg.sim.proc_file),
                        os.path.join(cfg.sim.path, cfg.sim.ocean_file),
                        os.path.join(cfg.sim.path, cfg.sim.xspectra_file))



    print('----------------------------------', flush=True)
    print(time.strftime("End of tasks [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
    print('----------------------------------', flush=True)


if __name__ == '__main__':

    # INPUT ARGUMENTS
    if len(sys.argv) < 2:
        dopscasim()
    else:
        dopscasim(sys.argv[1])


