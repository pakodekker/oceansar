#!/usr/bin/env python

""" ========================================
    sar_raw: SAR Raw data Generator (:mod:`srg`)
    ========================================

    Script to compute SAR Raw data from an ocean surface.

    **Arguments**
        * -c, --config_file: Configuration file
        * -o, --output_file: Output file
        * -oc, --ocean_file: Ocean output file
        * [-ro, --reuse_ocean_file]: Reuse ocean file if it exists
        * [-er, --errors_file]: Errors file (only if system errors model is activated)
        * [-re, --reuse_errors_file]: Reuse errors file if it exists

    .. note::
       This script MUST be run using MPI!

    Example (4 cores machine)::

        mpiexec -np 4 python sar_raw.py -c config.par -o raw_data.nc -oc ocean_state.nc -ro -er errors_file.nc -re

"""

from mpi4py import MPI

import time
import argparse

import numpy as np
from scipy import linalg
import numexpr as ne
import datetime

from oceansar import utils
from oceansar import io as tpio
from oceansar.utils import geometry as geosar
from oceansar.radarsim.antenna import sinc_1tx_nrx
from oceansar import constants as const

from oceansar import nrcs as rcs
from oceansar import closure
from oceansar.radarsim import range_profile as raw

from oceansar.surfaces import OceanSurface, OceanSurfaceBalancer


def sarraw(cfg_file, output_file, ocean_file, reuse_ocean_file, errors_file, reuse_errors_file):

    ###################
    # INITIALIZATIONS #
    ###################

    ## MPI SETUP
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()

    ## WELCOME
    if rank == 0:
        print('-------------------------------------------------------------------')
        print(time.strftime("- OCEANSAR SAR RAW GENERATOR: %Y-%m-%d %H:%M:%S", time.localtime()))
        print('- Copyright (c) Gerard Marull Paretas, Paco Lopez Dekker')
        print('-------------------------------------------------------------------')

    ## CONFIGURATION FILE
    # Note: variables are 'copied' to reduce code verbosity
    cfg = tpio.ConfigFile(cfg_file)

    # RAW
    wh_tol = cfg.srg.wh_tol
    nesz = cfg.srg.nesz
    use_hmtf = cfg.srg.use_hmtf
    scat_spec_enable = cfg.srg.scat_spec_enable
    scat_spec_mode = cfg.srg.scat_spec_mode
    scat_bragg_enable = cfg.srg.scat_bragg_enable
    scat_bragg_model = cfg.srg.scat_bragg_model
    scat_bragg_d = cfg.srg.scat_bragg_d
    scat_bragg_spec = cfg.srg.scat_bragg_spec
    scat_bragg_spread = cfg.srg.scat_bragg_spread

    # SAR
    inc_angle = np.deg2rad(cfg.sar.inc_angle)
    f0 = cfg.sar.f0
    pol = cfg.sar.pol
    squint_r = np.degrees(cfg.sar.squint)
    if pol == 'DP':
        do_hh = True
        do_vv = True
    elif pol == 'hh':
        do_hh = True
        do_vv = False
    else:
        do_hh = False
        do_vv = True

    prf = cfg.sar.prf
    num_ch = int(cfg.sar.num_ch)
    ant_L = cfg.sar.ant_L
    alt = cfg.sar.alt
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.sar.rg_bw
    over_fs = cfg.sar.over_fs
    sigma_n_tx = cfg.sar.sigma_n_tx
    phase_n_tx = np.deg2rad(cfg.sar.phase_n_tx)
    sigma_beta_tx = cfg.sar.sigma_beta_tx
    phase_beta_tx = np.deg2rad(cfg.sar.phase_beta_tx)
    sigma_n_rx = cfg.sar.sigma_n_rx
    phase_n_rx = np.deg2rad(cfg.sar.phase_n_rx)
    sigma_beta_rx = cfg.sar.sigma_beta_rx
    phase_beta_rx = np.deg2rad(cfg.sar.phase_beta_rx)

    # OCEAN / OTHERS
    ocean_dt = cfg.ocean.dt

    add_point_target = False
    use_numba = True
    n_sinc_samples = 8
    sinc_ovs = 20
    chan_sinc_vec = raw.calc_sinc_vec(n_sinc_samples, sinc_ovs, Fs=over_fs)
    # OCEAN SURFACE
    if rank == 0:
        print('Initializing ocean surface...')

        surface_full = OceanSurface()

        # Setup compute values
        compute = ['D', 'Diff', 'Diff2']
        if use_hmtf:
            compute.append('hMTF')

        # Try to reuse initialized surface
        if reuse_ocean_file:
            try:
                surface_full.load(ocean_file, compute)
            except RuntimeError:
                pass

        if (not reuse_ocean_file) or (not surface_full.initialized):

            if hasattr(cfg.ocean, 'use_buoy_data'):
                if cfg.ocean.use_buoy_data:
                    bdataf = cfg.ocean.buoy_data_file
                    date = datetime.datetime(np.int(cfg.ocean.year),
                                             np.int(cfg.ocean.month),
                                             np.int(cfg.ocean.day),
                                             np.int(cfg.ocean.hour),
                                             np.int(cfg.ocean.minute), 0)
                    date, bdata = tpio.load_buoydata(bdataf, date)
                    buoy_spec = tpio.BuoySpectra(bdata, heading=cfg.sar.heading, depth=cfg.ocean.depth)
                    dirspectrum_func = buoy_spec.Sk2
                    # Since the wind direction is included in the buoy data
                    wind_dir = 0
            else:
                dirspectrum_func = None
                wind_dir = np.deg2rad(cfg.ocean.wind_dir)

            surface_full.init(cfg.ocean.Lx, cfg.ocean.Ly, cfg.ocean.dx,
                              cfg.ocean.dy, cfg.ocean.cutoff_wl,
                              cfg.ocean.spec_model, cfg.ocean.spread_model,
                              wind_dir,
                              cfg.ocean.wind_fetch, cfg.ocean.wind_U,
                              cfg.ocean.current_mag,
                              np.deg2rad(cfg.ocean.current_dir),
                              cfg.ocean.swell_enable, cfg.ocean.swell_ampl,
                              np.deg2rad(cfg.ocean.swell_dir),
                              cfg.ocean.swell_wl,
                              compute, cfg.ocean.opt_res,
                              cfg.ocean.fft_max_prime,
                              choppy_enable=cfg.ocean.choppy_enable,
                              depth=cfg.ocean.depth,
                              dirspectrum_func=dirspectrum_func)

            surface_full.save(ocean_file)

    else:
        surface_full = None

    # Initialize surface balancer
    surface = OceanSurfaceBalancer(surface_full, ocean_dt)

    # CALCULATE PARAMETERS
    if rank == 0:
        print('Initializing simulation parameters...')

    # SR/GR/INC Matrixes
    sr0 = geosar.inc_to_sr(inc_angle, alt)
    gr0 = geosar.inc_to_gr(inc_angle, alt)
    gr = surface.x + gr0
    sr, inc, _ = geosar.gr_to_geo(gr, alt)
    sr -= np.min(sr)
    #inc = np.repeat(inc[np.newaxis, :], surface.Ny, axis=0)
    #sr = np.repeat(sr[np.newaxis, :], surface.Ny, axis=0)
    #gr = np.repeat(gr[np.newaxis, :], surface.Ny, axis=0)
    #Let's try to safe some memory and some operations
    inc = inc.reshape(1, inc.size)
    sr = sr.reshape(1, sr.size)
    gr = gr.reshape(1, gr.size)
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)

    # lambda, K, resolution, time, etc.
    l0 = const.c/f0
    k0 = 2.*np.pi*f0/const.c
    sr_res = const.c/(2.*rg_bw)
    if cfg.sar.L_total:
        ant_L = ant_L/np.float(num_ch)
        d_chan = ant_L
    else:
        if np.float(cfg.sar.Spacing) != 0:
            d_chan = np.float(cfg.sar.Spacing)
        else:
            d_chan = ant_L

    if v_ground == 'auto':
        v_ground = geosar.orbit_to_vel(alt, ground=True)
    t_step = 1./prf
    t_span = (1.5*(sr0*l0/ant_L) + surface.Ly)/v_ground
    az_steps = np.int(np.floor(t_span/t_step))

    # Number of RG samples
    max_sr = np.max(sr) + wh_tol + (np.max(surface.y_full) + (t_span/2.)*v_ground)**2./(2.*sr0)
    min_sr = np.min(sr) - wh_tol
    rg_samp_orig = np.int(np.ceil(((max_sr - min_sr)/sr_res)*over_fs))
    rg_samp = np.int(utils.optimize_fftsize(rg_samp_orig))

    # Other initializations
    if do_hh:
        proc_raw_hh = np.zeros([num_ch, az_steps, rg_samp], dtype=np.complex)
    if do_vv:
        proc_raw_vv = np.zeros([num_ch, az_steps, rg_samp], dtype=np.complex)
    t_last_rcs_bragg = -1.
    last_progress = -1
    NRCS_avg_vv = np.zeros(az_steps, dtype=np.float)
    NRCS_avg_hh = np.zeros(az_steps, dtype=np.float)


    ## RCS MODELS
    # Specular
    if scat_spec_enable:
        if scat_spec_mode == 'kodis':
            rcs_spec = rcs.RCSKodis(inc, k0, surface.dx, surface.dy)
        elif scat_spec_mode == 'fa' or scat_spec_mode == 'spa':
            spec_ph0 = np.random.uniform(0., 2.*np.pi,
                                         size=[surface.Ny, surface.Nx])
            rcs_spec = rcs.RCSKA(scat_spec_mode, k0, surface.x, surface.y,
                                 surface.dx, surface.dy)
        else:
            raise NotImplementedError('RCS mode %s for specular scattering not implemented' % scat_spec_mode)

    # Bragg
    if scat_bragg_enable:
        phase_bragg = np.zeros([2, surface.Ny, surface.Nx])
        bragg_scats = np.zeros([2, surface.Ny, surface.Nx], dtype=np.complex)
        # dop_phase_p = np.random.uniform(0., 2.*np.pi, size=[surface.Ny, surface.Nx])
        # dop_phase_m = np.random.uniform(0., 2.*np.pi, size=[surface.Ny, surface.Nx])
        tau_c = closure.grid_coherence(cfg.ocean.wind_U,surface.dx, f0)
        rndscat_p = closure.randomscat_ts(tau_c, (surface.Ny, surface.Nx), prf)
        rndscat_m = closure.randomscat_ts(tau_c, (surface.Ny, surface.Nx), prf)
        # NOTE: This ignores slope, may be changed
        k_b = 2.*k0*sin_inc
        c_b = sin_inc*np.sqrt(const.g/k_b + 0.072e-3*k_b)

        if scat_bragg_model == 'romeiser97':
            current_dir = np.deg2rad(cfg.ocean.current_dir)
            current_vec = (cfg.ocean.current_mag *
                           np.array([np.cos(current_dir),
                                     np.sin(current_dir)]))
            U_dir = np.deg2rad(cfg.ocean.wind_dir)
            U_vec = (cfg.ocean.wind_U *
                     np.array([np.cos(U_dir), np.sin(U_dir)]))
            U_eff_vec = U_vec - current_vec

            rcs_bragg = rcs.RCSRomeiser97(k0, inc, pol,
                                          surface.dx, surface.dy,
                                          linalg.norm(U_eff_vec),
                                          np.arctan2(U_eff_vec[1],
                                                     U_eff_vec[0]),
                                          surface.wind_fetch,
                                          scat_bragg_spec, scat_bragg_spread,
                                          scat_bragg_d)
        else:
            raise NotImplementedError('RCS model %s for Bragg scattering not implemented' % scat_bragg_model)

    surface_area = surface.dx * surface.dy * surface.Nx * surface.Ny
    ###################
    # SIMULATION LOOP #
    ###################
    if rank == 0:
        print('Computing profiles...')

    for az_step in np.arange(az_steps, dtype=np.int):

        ## AZIMUTH & SURFACE UPDATE
        t_now = az_step*t_step
        az_now = (t_now - t_span/2.)*v_ground
        # az = np.repeat((surface.y - az_now)[:, np.newaxis], surface.Nx, axis=1)
        az = (surface.y - az_now).reshape((surface.Ny, 1))
        surface.t = t_now


        ## COMPUTE RCS FOR EACH MODEL
        # Note: SAR processing is range independent as slant range is fixed
        sin_az = az / sr0
        az_proj_angle = np.arcsin(az / gr0)

        # Note: Projected displacements are added to slant range
        sr_surface = (sr - cos_inc*surface.Dz + az/2*sin_az
                         + surface.Dx*sin_inc + surface.Dy*sin_az)

        if do_hh:
            scene_hh = np.zeros([int(surface.Ny), int(surface.Nx)], dtype=np.complex)
        if do_vv:
            scene_vv = np.zeros([int(surface.Ny), int(surface.Nx)], dtype=np.complex)
        # Point target
        if add_point_target and rank == 0:
            sr_pt = (sr[0, surface.Nx/2] + az[surface.Ny/2, 0]/2 *
                     sin_az[surface.Ny/2, surface.Nx/2])
            pt_scat = (100. * np.exp(-1j * 2. * k0 * sr_pt))
            if do_hh:
                scene_hh[surface.Ny/2, surface.Nx/2] = pt_scat
            if do_vv:
                scene_vv[surface.Ny/2, surface.Nx/2] = pt_scat
            sr_surface[surface.Ny/2, surface.Nx/2] = sr_pt

        # Specular
        if scat_spec_enable:
            if scat_spec_mode == 'kodis':
                Esn_sp = np.sqrt(4.*np.pi)*rcs_spec.field(az_proj_angle, sr_surface,
                                                          surface.Diffx, surface.Diffy,
                                                          surface.Diffxx, surface.Diffyy, surface.Diffxy)
                if do_hh:
                    scene_hh += Esn_sp
                if do_vv:
                    scene_vv += Esn_sp
            else:
                # FIXME
                if do_hh:
                    pol_tmp = 'hh'
                    Esn_sp = (np.exp(-1j*(2.*k0*sr_surface)) * (4.*np.pi)**1.5 *
                              rcs_spec.field(1, 1, pol_tmp[0], pol_tmp[1],
                                             inc, inc,
                                             az_proj_angle, az_proj_angle + np.pi,
                                             surface.Dz,
                                             surface.Diffx, surface.Diffy,
                                             surface.Diffxx,
                                             surface.Diffyy,
                                             surface.Diffxy))
                    scene_hh += Esn_sp
                if do_vv:
                    pol_tmp = 'vv'
                    Esn_sp = (np.exp(-1j*(2.*k0*sr_surface)) * (4.*np.pi)**1.5 *
                              rcs_spec.field(1, 1, pol_tmp[0], pol_tmp[1],
                                             inc, inc,
                                             az_proj_angle, az_proj_angle + np.pi,
                                             surface.Dz,
                                             surface.Diffx, surface.Diffy,
                                             surface.Diffxx,
                                             surface.Diffyy,
                                             surface.Diffxy))
                    scene_vv += Esn_sp
            NRCS_avg_hh[az_step] += (np.sum(np.abs(Esn_sp)**2) / surface_area)
            NRCS_avg_vv[az_step] += NRCS_avg_hh[az_step]

        # Bragg
        if scat_bragg_enable:
            if (t_now - t_last_rcs_bragg) > ocean_dt:

                if scat_bragg_model == 'romeiser97':
                    if pol == 'DP':
                        RCS_bragg_hh, RCS_bragg_vv = rcs_bragg.rcs(az_proj_angle,
                                                                   surface.Diffx,
                                                                   surface.Diffy)
                    elif pol=='hh':
                        RCS_bragg_hh = rcs_bragg.rcs(az_proj_angle,
                                                     surface.Diffx,
                                                     surface.Diffy)
                    else:
                        RCS_bragg_vv = rcs_bragg.rcs(az_proj_angle,
                                                     surface.Diffx,
                                                     surface.Diffy)

                if use_hmtf:
                    # Fix Bad MTF points
                    surface.hMTF[np.where(surface.hMTF < -1)] = -1
                    if do_hh:
                        RCS_bragg_hh[0] *= (1 + surface.hMTF)
                        RCS_bragg_hh[1] *= (1 + surface.hMTF)
                    if do_vv:
                        RCS_bragg_vv[0] *= (1 + surface.hMTF)
                        RCS_bragg_vv[1] *= (1 + surface.hMTF)

                t_last_rcs_bragg = t_now

            if do_hh:
                scat_bragg_hh = np.sqrt(RCS_bragg_hh)
                NRCS_bragg_hh_instant_avg = np.sum(RCS_bragg_hh) / surface_area
                NRCS_avg_hh[az_step] += NRCS_bragg_hh_instant_avg
            if do_vv:
                scat_bragg_vv = np.sqrt(RCS_bragg_vv)
                NRCS_bragg_vv_instant_avg = np.sum(RCS_bragg_vv) / surface_area
                NRCS_avg_vv[az_step] += NRCS_bragg_vv_instant_avg


            # Doppler phases (Note: Bragg radial velocity taken constant!)
            surf_phase = - (2 * k0) * sr_surface
            cap_phase = (2 * k0) * t_step * c_b * (az_step + 1)
            phase_bragg[0] = surf_phase - cap_phase # + dop_phase_p
            phase_bragg[1] = surf_phase + cap_phase # + dop_phase_m
            bragg_scats[0] = rndscat_m.scats(t_now)
            bragg_scats[1] = rndscat_p.scats(t_now)
            if do_hh:
                scene_hh += ne.evaluate('sum(scat_bragg_hh * exp(1j*phase_bragg) * bragg_scats, axis=0)')
            if do_vv:
                scene_vv += ne.evaluate('sum(scat_bragg_vv * exp(1j*phase_bragg) * bragg_scats, axis=0)')

        ## ANTENNA PATTERN

        if cfg.sar.L_total:
            beam_pattern = sinc_1tx_nrx(sin_az, ant_L * num_ch, f0, num_ch, field=True)
        else:
            beam_pattern = sinc_1tx_nrx(sin_az, ant_L, f0, 1, field=True)

        ## GENERATE CHANEL PROFILES
        for ch in np.arange(num_ch, dtype=np.int):

            if do_hh:
                scene_bp = scene_hh * beam_pattern
                # Add channel phase & compute profile
                scene_bp *= np.exp(-1j*k0*d_chan*ch*sin_az)
                if use_numba:
                    raw.chan_profile_numba(sr_surface.flatten(),
                                           scene_bp.flatten(),
                                           sr_res/(over_fs),
                                           min_sr,
                                           chan_sinc_vec,
                                           n_sinc_samples, sinc_ovs,
                                           proc_raw_hh[ch][az_step])

                else:
                    raw.chan_profile_weave(sr_surface.flatten(),
                                           scene_bp.flatten(),
                                           sr_res/(over_fs),
                                           min_sr,
                                           chan_sinc_vec,
                                           n_sinc_samples, sinc_ovs,
                                           proc_raw_hh[ch][az_step])
            if do_vv:
                scene_bp = scene_vv * beam_pattern
                # Add channel phase & compute profile
                scene_bp *= np.exp(-1j*k0*d_chan*ch*sin_az)
                if use_numba:
                    raw.chan_profile_numba(sr_surface.flatten(),
                                           scene_bp.flatten(),
                                           sr_res/(over_fs),
                                           min_sr,
                                           chan_sinc_vec,
                                           n_sinc_samples, sinc_ovs,
                                           proc_raw_vv[ch][az_step])

                else:
                    raw.chan_profile_weave(sr_surface.flatten(),
                                           scene_bp.flatten(),
                                           sr_res/(over_fs),
                                           min_sr,
                                           chan_sinc_vec,
                                           n_sinc_samples, sinc_ovs,
                                           proc_raw_vv[ch][az_step])

        # SHOW PROGRESS (%)
        current_progress = np.int((100*az_step)/az_steps)
        if current_progress != last_progress:
            last_progress = current_progress
            print('SP,%d,%d,%d' % (rank, size, current_progress))


    # MERGE RESULTS
    if do_hh:
        total_raw_hh = np.empty_like(proc_raw_hh) if rank == 0 else None
        comm.Reduce(proc_raw_hh, total_raw_hh, op=MPI.SUM, root=0)
    if do_vv:
        total_raw_vv = np.empty_like(proc_raw_vv) if rank == 0 else None
        comm.Reduce(proc_raw_vv, total_raw_vv, op=MPI.SUM, root=0)

    ## PROCESS REDUCED RAW DATA & SAVE (ROOT)
    if rank == 0:
        print('Processing and saving results...')

        # Filter and decimate
        #range_filter = np.ones_like(total_raw)
        #range_filter[:, :, rg_samp/(2*2*cfg.sar.over_fs):-rg_samp/(2*2*cfg.sar.over_fs)] = 0

        #total_raw = np.fft.ifft(range_filter*np.fft.fft(total_raw))
        if do_hh:
            total_raw_hh = total_raw_hh[:, :, :rg_samp_orig]
        if do_vv:
            total_raw_vv = total_raw_vv[:, :, :rg_samp_orig]

        # Calibration factor (projected antenna pattern integrated in azimuth)
        az_axis = np.arange(-t_span/2.*v_ground, t_span/2.*v_ground, sr0*const.c/(np.pi*f0*ant_L*10.))

        if cfg.sar.L_total:
            pattern = sinc_1tx_nrx(az_axis/sr0, ant_L * num_ch, f0,
                                   num_ch, field=True)
        else:
            pattern = sinc_1tx_nrx(az_axis/sr0, ant_L, f0, 1,
                                   field=True)
        cal_factor = (1. / np.sqrt(np.trapz(np.abs(pattern)**2., az_axis) *
                      sr_res/np.sin(inc_angle)))

        if do_hh:
            noise = (utils.db2lin(nesz, amplitude=True) / np.sqrt(2.) *
                     (np.random.normal(size=total_raw_hh.shape) +
                      1j*np.random.normal(size=total_raw_hh.shape)))
            total_raw_hh = total_raw_hh * cal_factor + noise
        if do_vv:
            noise = (utils.db2lin(nesz, amplitude=True) / np.sqrt(2.) *
                     (np.random.normal(size=total_raw_vv.shape) +
                      1j*np.random.normal(size=total_raw_vv.shape)))
            total_raw_vv = total_raw_vv * cal_factor + noise

        # Add slow-time error
        # if use_errors:
        #     if do_hh:
        #         total_raw_hh *= errors.beta_noise
        #     if do_vv:
        #         total_raw_vv *= errors.beta_noise

        # Save RAW data (and other properties, used by 3rd party software)
        if do_hh and do_vv:
            rshp = (1,) + total_raw_hh.shape
            total_raw = np.concatenate((total_raw_hh.reshape(rshp),
                                        total_raw_vv.reshape(rshp)))
            rshp = (1,) + NRCS_avg_hh.shape
            NRCS_avg = np.concatenate((NRCS_avg_hh.reshape(rshp),
                                       NRCS_avg_vv.reshape(rshp)))
        elif do_hh:
            rshp = (1,) + total_raw_hh.shape
            total_raw = total_raw_hh.reshape(rshp)
            rshp = (1,) + NRCS_avg_hh.shape
            NRCS_avg = NRCS_avg_hh.reshape(rshp)
        else:
            rshp = (1,) + total_raw_vv.shape
            total_raw = total_raw_vv.reshape(rshp)
            rshp = (1,) + NRCS_avg_vv.shape
            NRCS_avg = NRCS_avg_vv.reshape(rshp)

        raw_file = tpio.RawFile(output_file, 'w', total_raw.shape)
        raw_file.set('inc_angle', np.rad2deg(inc_angle))
        raw_file.set('f0', f0)
        raw_file.set('num_ch', num_ch)
        raw_file.set('ant_L', ant_L)
        raw_file.set('prf', prf)
        raw_file.set('v_ground', v_ground)
        raw_file.set('orbit_alt', alt)
        raw_file.set('sr0', sr0)
        raw_file.set('rg_sampling', rg_bw*over_fs)
        raw_file.set('rg_bw', rg_bw)
        raw_file.set('raw_data*', total_raw)
        raw_file.set('NRCS_avg', NRCS_avg)
        raw_file.close()

        print(time.strftime("Finished [%Y-%m-%d %H:%M:%S]", time.localtime()))


if __name__ == '__main__':

    # INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_file')
    parser.add_argument('-o', '--output_file')
    parser.add_argument('-oc', '--ocean_file')
    parser.add_argument('-ro', '--reuse_ocean_file', action='store_true')
    parser.add_argument('-er', '--errors_file', type=str, default=None)
    parser.add_argument('-re', '--reuse_errors_file', action='store_true')
    args = parser.parse_args()

    sarraw(args.cfg_file, args.output_file,
           args.ocean_file, args.reuse_ocean_file,
           args.errors_file, args.reuse_errors_file)
