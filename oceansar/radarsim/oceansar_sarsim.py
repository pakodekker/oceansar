#!/usr/bin/python
# coding=utf-8

""" =============================================
    OCEANSAR SAR Simulation script
    =============================================


    **Arguments**
        - Configuration file

e.g. python oceansar_sarsim.py d:\\data\\configfile\\20160708_NOS.cfg

"""

import sys
import os
import time
#import stereoid.utils.config as st_config
from oceansar.memory import memory_limited
from oceansar import ocs_io as osrio
from drama import utils
from oceansar.radarsim.antenna_model import resolve_acquisition
from oceansar.radarsim.sar_system_errors import sar_system_errors, validate_raw_paths
from oceansar.sar_l1 import sar_focus
from oceansar.radarsim.sar_raw import sar_raw
from oceansar.radarsim.ati_processor import ati_process
from oceansar.radarsim.insar_processor import insar_process
from oceansar.radarsim.L2_wavespectrum import l2_wavespectrum

@memory_limited
def sarsim(cfg_file=None):

    """Orchestrate the configured SAR simulation and processing stages.

    Parameters
    ----------
    cfg_file : str or path-like, optional
        Configuration file selected by DRAMA's parameter-file helper. None
        opens a file-selection dialog. The sim section supplies the output
        directory, filenames and raw_run/proc_run/insar_run/ati_run/
        L2_wavespectrum_run switches.

    Returns
    -------
    None
        Copy the configuration to sim.path and write each enabled stage's
        products there: ideal raw data, ocean initial conditions, focused SLC,
        InSAR products, ATI reports/maps and wave cross-spectra as applicable.
        Stages disabled by the configuration require their input files to
        already exist when a downstream stage needs them.

    Notes
    -----
    Whenever proc_run is enabled, explicitly generate the system raw product,
    focus it, then delete it after successful focusing. The ideal raw file is
    retained. A failure propagates and leaves an intermediate that was created
    for inspection. The default temporary filename is raw_data_sys.nc.
    """
    print('-------------------------------------------------------------------', flush=True)
    print(time.strftime("OCEANSAR SAR Simulator [%Y-%m-%d %H:%M:%S]", time.localtime()), flush=True)
    # print('Copyright (c) Gerard Marull Paretas, Paco López-Dekker')
    print('-------------------------------------------------------------------', flush=True)
    cfg_file = utils.get_par_file(parfile=cfg_file)
    cfg = resolve_acquisition(osrio.ConfigFile(cfg_file))
    # Create output directory if it doesnt exist already
    os.makedirs(cfg.sim.path, exist_ok=True)
    cfg_copy = cfg.copy_to(cfg.sim.path)
    cfg.cfg_file_name = str(cfg_copy)
    print('Configuration bundle copied to %s' % cfg_copy, flush=True)

    # RAW
    if cfg.sim.raw_run:

        print('Launching SAR RAW Generator...')
        sar_raw(cfg.cfg_file_name,
                os.path.join(cfg.sim.path, cfg.sim.raw_file),
                os.path.join(cfg.sim.path, cfg.sim.ocean_file),
                cfg.sim.ocean_reuse, plot_save=True)

    # Processing
    if cfg.sim.proc_run:
        raw_file = os.path.join(cfg.sim.path, cfg.sim.raw_file)
        sys_raw_file = os.path.join(cfg.sim.path, getattr(cfg.sim, 'sys_raw_file', 'raw_data_sys.nc'))
        proc_file = os.path.join(cfg.sim.path, cfg.sim.proc_file)
        validate_raw_paths(raw_file, sys_raw_file, proc_file)

        print('Launching SAR System Errors Processor...')
        sar_system_errors(cfg.cfg_file_name, raw_file, sys_raw_file)

        print('Launching SAR Focusing Processor...')
        sar_focus(cfg.cfg_file_name, sys_raw_file, proc_file)

        # Keep the intermediate if a preceding stage raises an exception.
        os.remove(sys_raw_file)

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
        sarsim()
    else:
        sarsim(sys.argv[1])
