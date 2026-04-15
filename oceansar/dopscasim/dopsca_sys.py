import os
import numpy as np
import xarray as xr
from drama.io import cfg as drcfg
from drama import utils as drtls
import drama.geo as drgeo
import drama.performance.oscillators as drosc


def dopsca_syssim(cfg_file, output_file='auto', add_noise=True, clock_errors=True,plots=False):
    cfg = drcfg.ConfigFile(cfg_file)
    rawfile = os.path.join(cfg.sim.path, cfg.sim.ideal_raw_file)
    dtraw = xr.open_dataset(rawfile,engine='h5netcdf')
    dtraw.load()
    dtraw.close()
    geom = drgeo.QuickRadarGeometry(dtraw.orbit_alt.values)
    sr0 = dtraw.sr0.values
    sr = sr0 + 3e8 * np.arange(dtraw.rg_dim.size) / (2 * dtraw.rg_sampling.values)
    range_scaling = (sr0 / sr)**2
    craw = dtraw.raw_data_r.values + 1j * dtraw.raw_data_i.values
    dshape = craw.shape
    if add_noise:
        nesz = cfg.sca.nesz
        noise = (np.random.randn(*dshape) + 1j * np.random.randn(*dshape)) * 10**(nesz / 20) / np.sqrt(2)
        noise = noise * range_scaling[np.newaxis, np.newaxis, :]
        craw += noise

    if clock_errors:
        osc_coef = drosc.measured2coef(f=cfg.uso.f, S_meas=cfg.uso.Smeas, floor=cfg.uso.floor, plots=plots)
        print(osc_coef)
        # Phase Noise realization
        # total duration
        Tuso = dshape[1]/dtraw.prf.values + 1
        fs_uso = 2/dtraw.subpulse_length.values
        uso_ovs = int(dtraw.rg_sampling.values / fs_uso)
        fs_uso = dtraw.rg_sampling.values / uso_ovs
        pnoise, S2,f2 = drosc.phasenoise(Tuso, 1/fs_uso, osc_coef[0], osc_coef[1])
        # Phase noise is implicitly filtered
        for pulse in range(dshape[1]):
            # Noise, make it long enough
            s0 = int(pulse*fs_uso/dtraw.prf.values)
            pn = pnoise[s0:s0+int(dshape[2]/uso_ovs)+10]
            pn_rs = drtls.quadresample(pn,np.arange(dshape[2])/uso_ovs)
            pn_rf = pn_rs * dtraw.f0.values / cfg.uso.fref
            craw[:,pulse,:] = craw[:,pulse,:] * np.exp(1j*pn_rf[np.newaxis, :])
    
    # Add craw to dataset
    dtraw['craw'] = (dtraw['raw_data_r'].dims, craw)
    if output_file == 'auto':
        output_file = os.path.join(cfg.sim.path, cfg.sim.sys_raw_file)
    if not output_file is None:
        dtraw.to_netcdf(output_file,mode='w', engine='h5netcdf')
    
    return dtraw