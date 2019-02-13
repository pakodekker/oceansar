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
import os
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import numexpr as ne
import datetime

from oceansar import utils
from oceansar import ocs_io as tpio
from oceansar.utils import geometry as geosar
from oceansar.radarsim.antenna import sinc_1tx_nrx
from oceansar import constants as const

from oceansar import nrcs as rcs
from oceansar import closure
from oceansar.radarsim import range_profile as raw

from oceansar.surfaces import OceanSurface, OceanSurfaceBalancer
from oceansar.swell_spec import dir_swell_spec as s_spec


def upsample_and_dopplerize(ssraw, dop, n_up, prf):
    """

    :param ssraw: data
    :param dop: Geomeytric Dopper
    :param n_up: Upsampling factor
    :param prf: PRF
    :return:
    """
    # FIXME
    # We should add global (mean) range cell migration here
    # The varying part is handled in the main loop of the code
    dims = ssraw.shape
    print(n_up)
    out = np.zeros((dims[0] * int(n_up), dims[2]), dtype=np.complex64)
    # tmp = np.zeros([raw.shape[0] * int(n_up), raw.shape[2]], dtype=np.complex)
    t = (np.arange(dims[0] * int(n_up)) / prf).reshape((dims[0] * int(n_up), 1))
    t2pi = t * (np.pi * 2)
    for ind in range(dims[1]):
        raw_zp = np.zeros((dims[0] * int(n_up), dims[2]), dtype=np.complex64)
        raw_zp[0:dims[0]] = np.fft.fftshift(np.fft.fft(ssraw[:, ind, :], axis=0), axes=(0,))
        raw_zp = np.roll(raw_zp, int(-dims[0]/2), axis=0)
        raw_zp = np.conj(np.fft.fft(np.conj(raw_zp), axis=0)) / dims[0]
        dop_phase = t2pi * (dop[ind]).reshape((1, dims[2]))
        out = out + raw_zp * np.exp(1j * dop_phase)

    # out = np.zeros([ssraw.shape[0], int(n_up), ssraw.shape[2]], dtype=np.complex64)
    # out = out + np.sum(ssraw, axis=1).reshape((dims[0], 1, dims[2]))
    # out = out.reshape((ssraw.shape[0] * int(n_up), ssraw.shape[2]))
    return out


def skimraw(cfg_file, output_file, ocean_file, reuse_ocean_file, errors_file, reuse_errors_file,
            plot_save=True):

    ###################
    # INITIALIZATIONS #
    ###################

    # MPI SETUP
    comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()

    #  WELCOME
    if rank == 0:
        print('-------------------------------------------------------------------')
        print(time.strftime("- OCEANSAR SKIM RAW GENERATOR: %Y-%m-%d %H:%M:%S", time.localtime()))
        # print('- Copyright (c) Gerard Marull Paretas, Paco Lopez Dekker')
        print('-------------------------------------------------------------------')

    #  CONFIGURATION FILE
    # Note: variables are 'copied' to reduce code verbosity
    cfg = tpio.ConfigFile(cfg_file)
    info = utils.PrInfo(cfg.sim.verbosity, "SKIM raw")
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
    inc_angle = np.deg2rad(cfg.radar.inc_angle)
    f0 = cfg.radar.f0
    pol = cfg.radar.pol
    squint_r = np.radians(90 - cfg.radar.azimuth)
    if pol == 'DP':
        do_hh = True
        do_vv = True
    elif pol == 'hh':
        do_hh = True
        do_vv = False
    else:
        do_hh = False
        do_vv = True

    prf = cfg.radar.prf
    num_ch = int(cfg.radar.num_ch)
    ant_L = cfg.radar.ant_L
    alt = cfg.radar.alt
    v_ground = cfg.radar.v_ground
    rg_bw = cfg.radar.rg_bw
    over_fs = cfg.radar.Fs / cfg.radar.rg_bw
    sigma_n_tx = cfg.radar.sigma_n_tx
    phase_n_tx = np.deg2rad(cfg.radar.phase_n_tx)
    sigma_beta_tx = cfg.radar.sigma_beta_tx
    phase_beta_tx = np.deg2rad(cfg.radar.phase_beta_tx)
    sigma_n_rx = cfg.radar.sigma_n_rx
    phase_n_rx = np.deg2rad(cfg.radar.phase_n_rx)
    sigma_beta_rx = cfg.radar.sigma_beta_rx
    phase_beta_rx = np.deg2rad(cfg.radar.phase_beta_rx)

    # OCEAN / OTHERS
    ocean_dt = cfg.ocean.dt
    if hasattr(cfg.sim, "cal_targets"):
        if cfg.sim.cal_targets is False:
            add_point_target = False  # This for debugging
            point_target_floats = True  # Not really needed, but makes coding easier later
        else:
            print("Adding cal targets")
            add_point_target = True
            if cfg.sim.cal_targets.lower() == 'floating':
                point_target_floats = True
            else:
                point_target_floats = False
    else:
        add_point_target = False  # This for debugging
        point_target_floats = True

    n_sinc_samples = 10
    sinc_ovs = 20
    chan_sinc_vec = raw.calc_sinc_vec(n_sinc_samples, sinc_ovs, Fs=over_fs)
    # Set win direction with respect to beam
    # I hope the following line is correct, maybe sign is wrong
    wind_dir = cfg.radar.azimuth - cfg.ocean.wind_dir
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
                    # FIX-ME: direction needs to consider also azimuth of beam
                    buoy_spec = tpio.BuoySpectra(bdata, heading=cfg.radar.heading, depth=cfg.ocean.depth)
                    dirspectrum_func = buoy_spec.Sk2
                    # Since the wind direction is included in the buoy data
                    wind_dir = 0
                else:
                    dirspectrum_func = None
                    if cfg.ocean.swell_dir_enable:
                        dir_swell_spec = s_spec.ardhuin_swell_spec
                    else:
                        dir_swell_spec = None

                    wind_dir = np.deg2rad(wind_dir)
            else:
                if cfg.ocean.swell_dir_enable:
                    dir_swell_spec = s_spec.ardhuin_swell_spec
                else:
                    dir_swell_spec = None
                dirspectrum_func = None
                wind_dir = np.deg2rad(wind_dir)

            surface_full.init(cfg.ocean.Lx, cfg.ocean.Ly, cfg.ocean.dx,
                              cfg.ocean.dy, cfg.ocean.cutoff_wl,
                              cfg.ocean.spec_model, cfg.ocean.spread_model,
                              wind_dir,
                              cfg.ocean.wind_fetch, cfg.ocean.wind_U,
                              cfg.ocean.current_mag,
                              np.deg2rad(cfg.radar.azimuth - cfg.ocean.current_dir),
                              cfg.radar.azimuth - cfg.ocean.dir_swell_dir,
                              cfg.ocean.freq_r, cfg.ocean.sigf,
                              cfg.ocean.sigs, cfg.ocean.Hs,
                              cfg.ocean.swell_dir_enable,
                              cfg.ocean.swell_enable, cfg.ocean.swell_ampl,
                              np.deg2rad(cfg.radar.azimuth - cfg.ocean.swell_dir),
                              cfg.ocean.swell_wl,
                              compute, cfg.ocean.opt_res,
                              cfg.ocean.fft_max_prime,
                              choppy_enable=cfg.ocean.choppy_enable,
                              depth=cfg.ocean.depth,
                              dirspectrum_func=dirspectrum_func,
                              dir_swell_spec=dir_swell_spec)

            surface_full.save(ocean_file)
            # Now we plot the directional spectrum
            # self.wave_dirspec[good_k] = dirspectrum_func(self.kx[good_k], self.ky[good_k])
            plt.figure()
            plt.imshow(np.fft.fftshift(surface_full.wave_dirspec),
                       extent=[surface_full.kx.min(), surface_full.kx.max(),
                               surface_full.ky.min(), surface_full.ky.max()],
                       origin='lower',
                       cmap='inferno_r')

            plt.grid(True)
            pltax = plt.gca()
            pltax.set_xlim((-1, 1))
            pltax.set_ylim((-1, 1))
            Narr_length = 0.08 # np.min([surface_full.kx.max(), surface_full.ky.max()])
            pltax.arrow(0, 0,
                        -Narr_length * np.sin(np.radians(cfg.radar.heading)),
                        Narr_length * np.cos(np.radians(cfg.radar.heading)),
                        fc="k", ec="k")
            plt.xlabel('$k_x$ [rad/m]')
            plt.ylabel('$k_y$ [rad/m]')
            plt.colorbar()
            #plt.show()
            # Create plots directory
            plot_path = os.path.dirname(output_file) + os.sep + 'raw_plots'
            if plot_save:
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

            plt.savefig(os.path.join(plot_path, 'input_dirspectrum.png'))

            plt.close()

            if cfg.ocean.swell_dir_enable:
                plt.figure()
                plt.imshow(np.fft.fftshift(np.abs(surface_full.swell_dirspec)),
                           extent=[surface_full.kx.min(), surface_full.kx.max(),
                                   surface_full.ky.min(), surface_full.ky.max()],
                           origin='lower',
                           cmap='inferno_r')

                plt.grid(True)
                pltax = plt.gca()
                pltax.set_xlim((-0.1, 0.1))
                pltax.set_ylim((-0.1, 0.1))
                Narr_length = 0.08 # np.min([surface_full.kx.max(), surface_full.ky.max()])
                pltax.arrow(0, 0,
                            -Narr_length * np.sin(np.radians(cfg.radar.heading)),
                            Narr_length * np.cos(np.radians(cfg.radar.heading)),
                            fc="k", ec="k")
                plt.xlabel('$k_x$ [rad/m]')
                plt.ylabel('$k_y$ [rad/m]')
                plt.colorbar()
                #plt.show()
                # Create plots directory
                plot_path = os.path.dirname(output_file) + os.sep + 'raw_plots'
                if plot_save:
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)

                plt.savefig(os.path.join(plot_path, 'input_dirspectrum_combined.png'))

                plt.close()

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
    print(sr.dtype)
    look = geosar.inc_to_look(inc, alt)
    min_sr = np.min(sr)
    # sr -= np.min(sr)
    #inc = np.repeat(inc[np.newaxis, :], surface.Ny, axis=0)
    #sr = np.repeat(sr[np.newaxis, :], surface.Ny, axis=0)
    #gr = np.repeat(gr[np.newaxis, :], surface.Ny, axis=0)
    #Let's try to safe some memory and some operations
    inc = inc.reshape(1, inc.size)
    look = look.reshape(1, inc.size)
    sr = sr.reshape(1, sr.size)
    gr = gr.reshape(1, gr.size)
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)

    # lambda, K, resolution, time, etc.
    l0 = const.c/f0
    k0 = 2.*np.pi*f0/const.c
    sr_res = const.c/(2.*rg_bw)
    sr_smp = const.c / (2 * cfg.radar.Fs)
    if cfg.radar.L_total:
        ant_L = ant_L/np.float(num_ch)

    if v_ground == 'auto':
        v_ground = geosar.orbit_to_vel(alt, ground=True)
        v_orb = geosar.orbit_to_vel(alt, ground=False)
    else:
        v_orb = v_ground
    t_step = 1./prf
    az_steps = int(cfg.radar.n_pulses)
    t_span = az_steps / prf
    rg_samp = np.int(utils.optimize_fftsize(cfg.radar.n_rg))
    #min_sr = np.mean(sr) - rg_samp / 2 * sr_smp
    print(sr_smp)
    max_sr = np.mean(sr) + rg_samp / 2 * sr_smp
    sr_prof = (np.arange(rg_samp) - rg_samp/2) * sr_smp + np.mean(sr)
    gr_prof, inc_prof, look_prof, b_prof = geosar.sr_to_geo(sr_prof, alt)
    look_prof = look_prof.reshape((1, look_prof.size))
    sr_prof = sr_prof.reshape((1, look_prof.size))
    dop_ref = 2 * v_orb * np.sin(look_prof) * np.sin(squint_r) / l0
    print("skim_raw: Doppler Centroid is %f Hz" % (np.mean(dop_ref)))
    if cfg.srg.two_scale_Doppler:
        # We will compute less surface realizations
        n_pulses_b = utils.optimize_fftsize(int(cfg.srg.surface_coh_time * prf))/2
        print("skim_raw: down-sampling rate =%i" % (n_pulses_b))
        # n_pulses_b = 4
        az_steps_ = int(np.ceil(az_steps / n_pulses_b))
        t_step = t_step * n_pulses_b
        # Maximum length in azimuth that we can consider to have the same geometric Doppler
        dy_integ = cfg.srg.phase_err_tol / (2 * k0 * v_ground / min_sr * cfg.srg.surface_coh_time)
        surface_dy = surface.y[1] - surface.y[0]
        ny_integ = (dy_integ / surface.dy)
        print("skim_raw: ny_integ=%f" % (ny_integ))
        if ny_integ < 1:
            ny_integ = 1
        else:
            ny_integ = int(2**np.floor(np.log2(ny_integ)))
        info.msg("skim_raw: size of intermediate radar data: %f MB" % (8 * ny_integ * az_steps_ * rg_samp *1e-6),
                 importance=1)
        info.msg("skim_raw: ny_integ=%i" % (ny_integ), importance=1)
        if do_hh:
            proc_raw_hh = np.zeros([az_steps_, int(surface.Ny / ny_integ), rg_samp], dtype=np.complex)
            proc_raw_hh_step = np.zeros([surface.Ny, rg_samp], dtype=np.complex)
        if do_vv:
            proc_raw_vv = np.zeros([az_steps_, int(surface.Ny / ny_integ), rg_samp], dtype=np.complex)
            proc_raw_vv_step = np.zeros([surface.Ny, rg_samp], dtype=np.complex)
        # Doppler centroid
        # sin(a+da) = sin(a) + cos(a)*da - 1/2*sin(a)*da**2
        az = surface.y.reshape((surface.Ny, 1))
        # FIX-ME: this is a coarse approximation
        da = az/gr_prof
        sin_az = np.sin(squint_r) + np.cos(squint_r) * da - 0.5 * np.sin(squint_r) * da**2
        dop0 = 2 * v_orb * np.sin(look_prof) * sin_az / l0
        # print("Max az: %f" % (np.max(az)))
        #dop0 = np.mean(np.reshape(dop0, (surface.Ny/ny_integ, ny_integ, rg_samp)), axis=1)
        s_int = np.int(surface.Ny / ny_integ)
        dop0 = np.mean(np.reshape(dop0, (s_int, np.int(ny_integ), rg_samp)), axis=1)
    else:
        az_steps_ = az_steps
        if do_hh:
            proc_raw_hh = np.zeros([az_steps, rg_samp], dtype=np.complex)
        if do_vv:
            proc_raw_vv = np.zeros([az_steps, rg_samp], dtype=np.complex)

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

    for az_step in np.arange(az_steps_, dtype=np.int):

        # AZIMUTH & SURFACE UPDATE
        t_now = az_step * t_step
        az_now = (t_now - t_span/2.)*v_ground * np.cos(squint_r)
        # az = np.repeat((surface.y - az_now)[:, np.newaxis], surface.Nx, axis=1)
        az = (surface.y - az_now).reshape((surface.Ny, 1))
        surface.t = t_now
        if az_step == 0:
            # Check wave-height
            info.msg("Standard deviation of wave-height (peak-to-peak; i.e. x2): %f" % (2 * np.std(surface.Dz)))
        #if az_step == 0:
        # print("Max Dx: %f" % (np.max(surface.Dx)))
        # print("Max Dy: %f" % (np.max(surface.Dy)))
        # print("Max Dz: %f" % (np.max(surface.Dz)))
        # print("Max Diffx: %f" % (np.max(surface.Diffx)))
        # print("Max Diffy: %f" % (np.max(surface.Diffy)))
        # print("Max Diffxx: %f" % (np.max(surface.Diffxx)))
        # print("Max Diffyy: %f" % (np.max(surface.Diffyy)))
        # print("Max Diffxy: %f" % (np.max(surface.Diffxy)))
        # COMPUTE RCS FOR EACH MODEL
        # Note: SAR processing is range independent as slant range is fixed
        sin_az = az / sr
        az_proj_angle = np.arcsin(az / gr0)
        # Note: Projected displacements are added to slant range
        if point_target_floats is False:  # This can only happen if point targets are enabled
            surface.Dx[int(surface.Ny / 2), int(surface.Nx / 2)] = 0
            surface.Dy[int(surface.Ny / 2), int(surface.Nx / 2)] = 0
            surface.Dz[int(surface.Ny / 2), int(surface.Nx / 2)] = 0

        if cfg.srg.two_scale_Doppler:
            # slant-range for phase
            sr_surface = (sr - cos_inc * surface.Dz
                          + surface.Dx * sin_inc + surface.Dy * sin_az)
            if cfg.srg.rcm:
                # add non common rcm
                sr_surface4rcm = sr_surface + az / 2 * sin_az
            else:
                sr_surface4rcm = sr_surface
        else:
            # FIXME: check if global shift is included, in case we care about slow simulations
            # slant-range for phase and Doppler
            sr_surface = (sr - cos_inc*surface.Dz + az/2*sin_az
                          + surface.Dx*sin_inc + surface.Dy*sin_az)
            sr_surface4rcm = sr_surface

        if do_hh:
            scene_hh = np.zeros([int(surface.Ny), int(surface.Nx)], dtype=np.complex)
        if do_vv:
            scene_vv = np.zeros([int(surface.Ny), int(surface.Nx)], dtype=np.complex)


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

        if add_point_target:
            # Now we replace scattering at center by fixed value
            pt_y = int(surface.Ny / 2)
            pt_x = int(surface.Nx / 2)
            if do_hh:
                scene_hh[pt_y, pt_x] = 1000 * np.exp(-1j * 2 * k0 * sr_surface[pt_y, pt_x])
            if do_vv:
                scene_vv[pt_y, pt_x] = 1000 * np.exp(-1j * 2 * k0 * sr_surface[pt_y, pt_x])
        ## ANTENNA PATTERN
        ## FIXME: this assume co-located Tx and Tx, so it will not work for true bistatic configurations
        if cfg.radar.L_total:
            beam_pattern = sinc_1tx_nrx(sin_az, ant_L * num_ch, f0, num_ch, field=True)
        else:
            beam_pattern = sinc_1tx_nrx(sin_az, ant_L, f0, 1, field=True)

        # GENERATE CHANEL PROFILES
        if cfg.srg.two_scale_Doppler:
            sr_surface_ = sr_surface4rcm
            if do_hh:
                proc_raw_hh_step[:, :] = 0
                proc_raw_hh_ = proc_raw_hh_step
                scene_bp_hh = scene_hh * beam_pattern
            if do_vv:
                proc_raw_vv_step[:, :] = 0
                proc_raw_vv_ = proc_raw_vv_step
                scene_bp_vv = scene_vv * beam_pattern
        else:
            sr_surface_ = sr_surface4rcm.flatten()
            if do_hh:
                proc_raw_hh_ = proc_raw_hh[az_step]
                scene_bp_hh = (scene_hh * beam_pattern).flatten()
            if do_vv:
                proc_raw_vv_ = proc_raw_vv[az_step]
                scene_bp_vv = (scene_vv * beam_pattern).flatten()
        if do_hh:
            raw.chan_profile_numba(sr_surface_,
                                   scene_bp_hh,
                                   sr_smp,
                                   sr_prof.min(),
                                   chan_sinc_vec,
                                   n_sinc_samples, sinc_ovs,
                                   proc_raw_hh_,
                                   rg_only=cfg.srg.two_scale_Doppler)
        if do_vv:
            raw.chan_profile_numba(sr_surface_,
                                   scene_bp_vv,
                                   sr_smp,
                                   sr_prof.min(),
                                   chan_sinc_vec,
                                   n_sinc_samples, sinc_ovs,
                                   proc_raw_vv_,
                                   rg_only=cfg.srg.two_scale_Doppler)
        if cfg.srg.two_scale_Doppler:
            #Integrate in azimuth
            s_int = np.int(surface.Ny/ny_integ)
            if do_hh:
                proc_raw_hh[az_step] = np.sum(np.reshape(proc_raw_hh_,
                                                         (s_int, ny_integ, rg_samp)), axis=1)
                info.msg("Max abs(HH): %f" % np.max(np.abs(proc_raw_hh[az_step])), importance=1)
            if do_vv:
                #print(proc_raw_vv.shape)
                proc_raw_vv[az_step] = np.sum(np.reshape(proc_raw_vv_,
                                                         (s_int, ny_integ, rg_samp)), axis=1)
                info.msg("Max abs(VV): %f" % np.max(np.abs(proc_raw_vv[az_step])), importance=1)
        # SHOW PROGRESS (%)
        current_progress = np.int((100*az_step)/az_steps_)
        if current_progress != last_progress:
            last_progress = current_progress
            info.msg('SP,%d,%d,%d%%' % (rank, size, current_progress), importance=1)

    if cfg.srg.two_scale_Doppler:
        # No we have to up-sample and add Doppler
        info.msg("skim_raw: Dopplerizing and upsampling")
        print(dop0.max())
        print(n_pulses_b)
        print(prf)
        if do_hh:
            proc_raw_hh = upsample_and_dopplerize(proc_raw_hh, dop0, n_pulses_b, prf)
        if do_vv:
            proc_raw_vv = upsample_and_dopplerize(proc_raw_vv, dop0, n_pulses_b, prf)

    # MERGE RESULTS
    if do_hh:
        total_raw_hh = np.empty_like(proc_raw_hh) if rank == 0 else None
        comm.Reduce(proc_raw_hh, total_raw_hh, op=MPI.SUM, root=0)
    if do_vv:
        total_raw_vv = np.empty_like(proc_raw_vv) if rank == 0 else None
        comm.Reduce(proc_raw_vv, total_raw_vv, op=MPI.SUM, root=0)

    ## PROCESS REDUCED RAW DATA & SAVE (ROOT)
    if rank == 0:
        info.msg('calibrating and saving results...')

        # Filter and decimate
        #range_filter = np.ones_like(total_raw)
        #range_filter[:, :, rg_samp/(2*2*cfg.radar.over_fs):-rg_samp/(2*2*cfg.radar.over_fs)] = 0

        #total_raw = np.fft.ifft(range_filter*np.fft.fft(total_raw))
        if do_hh:
            total_raw_hh = total_raw_hh[:, :cfg.radar.n_rg]
        if do_vv:
            total_raw_vv = total_raw_vv[:, :cfg.radar.n_rg]

        # Calibration factor (projected antenna pattern integrated in azimuth)
        az_axis = np.arange(-t_span/2.*v_ground, t_span/2.*v_ground, sr0*const.c/(np.pi*f0*ant_L*10.))

        if cfg.radar.L_total:
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

        # Save RAW data
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

        raw_file = tpio.SkimRawFile(output_file, 'w', total_raw.shape)
        raw_file.set('inc_angle', np.rad2deg(inc_angle))
        raw_file.set('f0', f0)
        # raw_file.set('num_ch', num_ch)
        raw_file.set('ant_L', ant_L)
        raw_file.set('prf', prf)
        raw_file.set('v_ground', v_ground)
        raw_file.set('orbit_alt', alt)
        raw_file.set('sr0', sr0)
        raw_file.set('rg_sampling', rg_bw*over_fs)
        raw_file.set('rg_bw', rg_bw)
        raw_file.set('raw_data*', total_raw)
        raw_file.set('NRCS_avg', NRCS_avg)
        raw_file.set('azimuth', cfg.radar.azimuth)
        raw_file.set('dop_ref', dop_ref)
        raw_file.close()

        print(time.strftime("Finished [%Y-%m-%d %H:%M:%S]", time.localtime()))


if __name__ == '__main__':

    # INPUT ARGUMENTS
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-c', '--cfg_file')
#    parser.add_argument('-o', '--output_file')
#    parser.add_argument('-oc', '--ocean_file')
#    parser.add_argument('-ro', '--reuse_ocean_file', action='store_true')
#    parser.add_argument('-er', '--errors_file', type=str, default=None)
#    parser.add_argument('-re', '--reuse_errors_file', action='store_true')
#    args = parser.parse_args()
#
#    skimraw(args.cfg_file, args.output_file,
#           args.ocean_file, args.reuse_ocean_file,
#           args.errors_file, args.reuse_errors_file)
##
    skimraw(r"D:\research\TU Delft\Data\OceanSAR\SKIM_proxy_new.cfg",
            r"D:\research\TU Delft\Data\OceanSAR\out1.nc",
           r"D:\research\TU Delft\Data\OceanSAR\out2.nc",
           False,
           False,
           False)


