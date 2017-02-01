#!/usr/bin/python

""" =============================================
    OCEANSAR SAR Simulation script
    =============================================


    **Arguments**
        - Configuration file

"""

import sys
import os
import time
import subprocess
from oceansar import io as osrio
from oceansar import utils

def sarsim(cfg_file=None):

    print('-------------------------------------------------------------------')
    print(time.strftime("OCEANSAR SAR Simulator [%Y-%m-%d %H:%M:%S]", time.localtime()))
    #print('Copyright (c) Gerard Marull Paretas, Paco López-Dekker')
    print('-------------------------------------------------------------------',
          flush=True)
    cfg_file = utils.get_parFile(parfile=cfg_file)
    cfg = osrio.ConfigFile(cfg_file)
    # Create output directory if it doesnt exist already
    os.makedirs(cfg.sim.path, exist_ok=True)
    src_path = os.path.dirname(os.path.abspath(__file__))

    # RAW
    if cfg.sim.raw_run:

        print('Launching SAR RAW Generator...', flush=True)

        args = [cfg.sim.mpi_exec,
                '-np', str(cfg.sim.mpi_num_proc),
                sys.executable, src_path + os.sep + 'sar_raw.py',
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
        print('Launching SAR RAW Processor...', flush=True)

        returncode = subprocess.call([sys.executable,
                                      src_path + os.sep + 'sar_processor.py',
                                      '-c', cfg.cfg_file_name,
                                      '-r', cfg.sim.path + os.sep + cfg.sim.raw_file,
                                      '-o', cfg.sim.path + os.sep + cfg.sim.proc_file])

        if returncode != 0:
            raise Exception('Something went wrong with SAR RAW Processor (return code %d)...' % returncode)

    # ATI
    if cfg.sim.corar_run:
        print('CoRAR Processing')

    if cfg.sim.ati_run:
        print('Launching SAR ATI Processor...', flush=True)

        returncode = subprocess.call([sys.executable,
                                      src_path + os.sep + 'sap.py',
                                      '-c', cfg.cfg_file_name,
                                      '-p', cfg.sim.path + os.sep + cfg.sim.proc_file,
                                      '-s', cfg.sim.path + os.sep + cfg.sim.ocean_file,
                                      '-o', cfg.sim.path + os.sep + cfg.sim.ati_file])

        if returncode != 0:
            raise Exception('Something went wrong with SAR ATI Processor (return code %d)...' % returncode)


    print('----------------------------------')
    print(time.strftime("End of tasks [%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('----------------------------------')


if __name__ == '__main__':

    # INPUT ARGUMENTS
    if len(sys.argv) < 2:
        sarsim()
    else:
        sarsim(sys.argv[1])
