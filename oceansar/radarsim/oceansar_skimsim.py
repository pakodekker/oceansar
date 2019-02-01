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
import argparse


def skimsim(cfg_file=None):

    print('-------------------------------------------------------------------', flush=True)
    print(time.strftime("OCEANSAR SKIM Simulator [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
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

        args = [cfg.sim.mpi_exec,
                '-np', str(cfg.sim.mpi_num_proc),
                sys.executable, src_path + os.sep + 'skim_raw.py',
                '-c', cfg.cfg_file_name,
                '-o', cfg.sim.path + os.sep + cfg.sim.raw_file,
                '-oc', cfg.sim.path + os.sep + cfg.sim.ocean_file,
                '-er', cfg.sim.path + os.sep + cfg.sim.errors_file]

        if cfg.sim.ocean_reuse:
            args.append('-ro')
        if cfg.sim.errors_reuse:
            args.append('-re')

        returncode = subprocess.call(args)

        if returncode != 0:
            raise Exception('Something went wrong with SAR RAW Generator (return code %d)...' % returncode)

    # Processing
    if cfg.sim.proc_run:
        print('Launching SAR RAW Processor...')

        returncode = subprocess.call([sys.executable,
                                      src_path + os.sep + 'skim_processor.py',
                                      '-c', cfg.cfg_file_name,
                                      '-r', cfg.sim.path + os.sep + cfg.sim.raw_file])
        #,
        #                              '-o', cfg.sim.path + os.sep + cfg.sim.pp_file])

        if returncode != 0:
            raise Exception('Something went wrong with SAR RAW Processor (return code %d)...' % returncode)



    print('----------------------------------', flush=True)
    print(time.strftime("End of tasks [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
    print('----------------------------------', flush=True)


if __name__ == '__main__':

    # INPUT ARGUMENTS
    #if len(sys.argv) < 2:
    #    skimsim()
    #else:
    #    skimsim(sys.argv[1])
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file_name')
    args = parser.parse_args()
    skimsim(args.cfg_file_name)
