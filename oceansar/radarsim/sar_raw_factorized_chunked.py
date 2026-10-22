"""Factorized raw generation with bounded azimuth workspaces.

The public sar_raw entry point selects this backend for Bragg/Kodis scenes.
The independent unchunked path remains available for numerical comparisons
and KA specular modes that do not yet support spatial chunks.
"""

import os
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.integrate import trapezoid
import numexpr as ne
import datetime

from tqdm import tqdm

from drama import utils as drtls
from drama.geo.geo_history import GeoHistory
from drama.performance.sar.sar_performance_common import calc_analysis_time
from oceansar.memory import memory_limited
from oceansar import utils
from oceansar import ocs_io as tpio
from oceansar.ocs_io import product_metadata as metadata
from oceansar.utils import geometry as geosar
from oceansar.radarsim.antenna_model import SarAntenna, resolve_acquisition, channel_positions
from oceansar import constants as const

from oceansar import nrcs as rcs
from oceansar import closure
from oceansar.radarsim import range_profile as raw

from oceansar.surfaces import OceanSurface
from oceansar.swell_spec import dir_swell_spec as s_spec
from oceansar.radarsim.factorize_raw import (
    aggregate_factorized_raw,
    factorize_raw_params,
    factorization_scene_time,
    factorization_current_range,
    nominal_raw_time_span,
)


@memory_limited
def sar_raw_factorized_chunked(cfg_file, output_file, ocean_file, reuse_ocean_file=False,
                               plot_save=True, bistatic=False, dau=0, *,
                               blocks_per_chunk=16, factorization=None, overwrite=False):
    """Generate a factorized raw product using bounded azimuth workspaces.

    Parameters
    ----------
    cfg_file : str or path-like
        Simulator configuration. srg.factorize must be True. Supports Romeiser
        Bragg scattering, Kodis specular scattering and point targets.
    output_file : str or path-like
        Ideal, uncalibrated RawFile destination. Must be fresh unless overwrite
        is True. Variables, calibration, geometry and sampling follow sar_raw;
        chunk provenance is added.
    ocean_file : str or path-like
        Ocean initial-condition file to create or reuse.
    reuse_ocean_file : bool, optional
        Load ocean_file instead of creating another realization when possible.
    plot_save : bool, optional
        Retain the reference generator's diagnostic plotting behaviour.
    bistatic : bool, optional
        Pass bistatic geometry to GeoHistory; the same quasi-monostatic
        scattering/antenna approximations as sar_raw still apply.
    dau : float, optional
        Along-track spacecraft displacement in metres.
    blocks_per_chunk : int, optional
        Number of existing factorization blocks in each spatial work group.
        Defaults to 16. The last group may be smaller; the physical block
        sizes, centres, coarse times and interpolation remain unchanged.
    factorization : object, optional
        Same parameters/aggregate strategy interface as sar_raw, allowing
        validation of the coarse profiles before unchanged reconstruction.

    overwrite : bool, optional
        Allow replacing an existing raw output. Defaults to False for direct
        backend/test calls. Public sar_raw passes True to preserve its normal
        rerun behaviour; existing data are not removed before generation.

    Returns
    -------
    None
        Write ideal raw samples, metadata and diagnostics to disk. System
        errors and focusing remain explicit later orchestration steps.

    Notes
    -----
    Public sar_raw selects this implementation from srg.factorize. Initialization
    and writing originated in the unchunked backend at commit 2d6f7f4; propagate
    common fixes to both and preserve the independent equivalence checks.
    Full ocean FFTs and Bragg RCS update/cache timing are retained. Only
    pointwise spatial work and range projection are chunked. Large absolute
    ranges remain float64 through origin subtraction and phase formation.
    The Numba range-profile creator and reconstruction are unchanged.
    sim.memory_limit_gib applies through the same monitored-worker decorator.
    """
    if (isinstance(blocks_per_chunk, (bool, np.bool_))
            or not isinstance(blocks_per_chunk, (int, np.integer))
            or blocks_per_chunk <= 0):
        raise ValueError('blocks_per_chunk must be a positive integer')
    if not isinstance(overwrite, (bool, np.bool_)):
        raise ValueError('overwrite must be a boolean')
    if not overwrite and os.path.exists(output_file):
        raise FileExistsError(output_file)

    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR RAW GENERATOR: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('- Copyright (c) Gerard Marull Paretas, Paco Lopez Dekker')
    print('-------------------------------------------------------------------')

    ## CONFIGURATION FILE
    # Note: variables are 'copied' to reduce code verbosity
    cfg = resolve_acquisition(tpio.ConfigFile(cfg_file))
    if not cfg.srg.factorize:
        raise ValueError('Chunked generator requires srg.factorize=True')
    if cfg.srg.scat_spec_enable and cfg.srg.scat_spec_mode != 'kodis':
        raise NotImplementedError('Chunked specular generation currently supports only kodis')
    antennas = SarAntenna.from_config(cfg)
    b_ati, b_xti = channel_positions(cfg)
    has_cross_track_baseline = bool(np.any(b_xti != 0.0))
    seed = utils.get_sim_seed(cfg)
    surface_rng = utils.random_generator(seed, 'surface')
    bragg_positive_rng = utils.random_generator(seed, 'bragg_positive')
    bragg_negative_rng = utils.random_generator(seed, 'bragg_negative')
    info = drtls.PrInfo(cfg.sim.verbosity, "SAR raw")
    # RAW
    wh_tol = cfg.srg.wh_tol
    use_hmtf = cfg.srg.use_hmtf
    scat_spec_enable = cfg.srg.scat_spec_enable
    scat_bragg_enable = cfg.srg.scat_bragg_enable
    scat_bragg_model = cfg.srg.scat_bragg_model
    scat_bragg_d = cfg.srg.scat_bragg_d
    scat_bragg_spec = cfg.srg.scat_bragg_spec
    scat_bragg_spread = cfg.srg.scat_bragg_spread
    use_lut = cfg.srg.use_lut
    simpar = {}
    # SAR
    inc_angle = np.deg2rad(cfg.mode.inc_angle)
    f0 = cfg.sar.f0
    pol = cfg.mode.pol
    simpar['pol'] = pol
    simpar['f0'] = f0
    if pol == 'DP':
        do_hh = True
        do_vv = True
    elif pol == 'hh':
        do_hh = True
        do_vv = False
    else:
        do_hh = False
        do_vv = True

    prf = cfg.mode.prf
    simpar['prf'] = prf
    num_ch = int(cfg.sar.num_ch)
    #ant_L = cfg.sar.ant_L
    alt = cfg.orbit.Horb
    v_ground = cfg.sar.v_ground
    rg_bw = cfg.mode.rg_bw
    over_fs = cfg.mode.over_fs
    simpar['Fs'] = rg_bw * over_fs

    # OCEAN / OTHERS
    ocean_dt = cfg.ocean.dt
    if hasattr(cfg.sim, 'add_point_target'):
        add_point_target = cfg.sim.add_point_target
    else:
        add_point_target = False

    n_sinc_samples = 8
    sinc_ovs = 20
    chan_sinc_vec = raw.calc_sinc_vec(n_sinc_samples, sinc_ovs, Fs=over_fs)
    # OCEAN SURFACE

    info.msg('Initializing ocean surface...', importance=2)

    surface = OceanSurface()

    # Setup compute values
    compute = ['D', 'Diff', 'Diff2'] if scat_spec_enable else ['D', 'Diff']

    if use_hmtf:
        compute.append('hMTF')

    # Try to reuse initialized surface
    if reuse_ocean_file:
        try:
            surface.load(ocean_file, compute)
        except RuntimeError:
            pass

    if (not reuse_ocean_file) or (not surface.initialized):
        if hasattr(cfg.ocean, 'use_buoy_data'):
            if cfg.ocean.use_buoy_data:
                bdataf = cfg.ocean.buoy_data_file
                date = datetime.datetime(int(cfg.ocean.year),
                                         int(cfg.ocean.month),
                                         int(cfg.ocean.day),
                                         int(cfg.ocean.hour),
                                         int(cfg.ocean.minute), 0)
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

                wind_dir = np.deg2rad(cfg.ocean.wind_dir)
        else:
            if cfg.ocean.swell_dir_enable:
                dir_swell_spec = s_spec.ardhuin_swell_spec
            else:
                dir_swell_spec = None
            dirspectrum_func = None
            wind_dir = np.deg2rad(cfg.ocean.wind_dir)

        surface.init(cfg.ocean.Lx, cfg.ocean.Ly, cfg.ocean.dx,
                     cfg.ocean.dy, cfg.ocean.cutoff_wl,
                     cfg.ocean.spec_model, cfg.ocean.spread_model,
                     wind_dir,
                     cfg.ocean.wind_fetch, cfg.ocean.wind_U,
                     cfg.ocean.current_mag,
                     np.deg2rad(cfg.ocean.current_dir),
                     cfg.ocean.dir_swell_dir,
                     cfg.ocean.freq_r, cfg.ocean.sigf,
                     cfg.ocean.sigs, cfg.ocean.Hs,
                     cfg.ocean.swell_dir_enable,
                     cfg.ocean.swell_enable, cfg.ocean.swell_ampl,
                     np.deg2rad(cfg.ocean.swell_dir),
                     cfg.ocean.swell_wl,
                     compute, cfg.ocean.opt_res,
                     cfg.ocean.fft_max_prime,
                     choppy_enable=cfg.ocean.choppy_enable,
                     depth=cfg.ocean.depth,
                     dirspectrum_func=dirspectrum_func,
                     dir_swell_spec=dir_swell_spec,
                     workers=cfg.sim.nworkers,
                     rng=surface_rng)

        surface.save(ocean_file)

        # Now we plot the directional spectrum
        # self.wave_dirspec[good_k] = dirspectrum_func(self.kx[good_k], self.ky[good_k])
        plt.figure()
        plt.imshow(np.fft.fftshift(surface.wave_dirspec),
                   extent=[surface.kx.min(), surface.kx.max(),
                           surface.ky.min(), surface.ky.max()],
                   origin='lower',
                   cmap='inferno_r')

        plt.grid(True)
        pltax = plt.gca()
        pltax.set_xlim((-0.1, 0.1))
        pltax.set_ylim((-0.1, 0.1))
        Narr_length = 0.08 # np.min([surface.kx.max(), surface.ky.max()])
        pltax.arrow(0, 0,
                    -Narr_length * np.sin(np.radians(cfg.sar.heading)),
                    Narr_length * np.cos(np.radians(cfg.sar.heading)),
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

    # CALCULATE PARAMETERS

    info.msg('Initializing simulation parameters...', importance=2)


    # SR/GR/INC Matrixes
    sr0 = geosar.inc_to_sr(inc_angle, alt)
    gr0 = geosar.inc_to_gr(inc_angle, alt)
    simpar['sr0'] = sr0
    simpar['gr0'] = gr0
    gr = surface.x + gr0
    sr, inc, la = geosar.gr_to_geo(gr, alt)
    # Slant range of first range gate
    # We have to correct one sample delay introduced in the range profile creator
    rg_sampling = rg_bw * over_fs
    simpar['rg_sampling'] = rg_sampling
    sr_near = sr[0] - wh_tol + const.c / 2 / (rg_sampling)
    simpar['sr_near'] = sr_near
    sr -= np.min(sr)

    # Let's try to safe some memory and some operations
    inc = inc.reshape(1, inc.size)
    sr = sr.reshape(1, sr.size)
    gr = gr.reshape(1, gr.size)
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)

    # lambda, K, resolution, time, etc.
    l0 = const.c/f0
    k0 = 2.*np.pi*f0/const.c
    sr_res = const.c/(2.*rg_bw)
    simpar['sr_res'] = sr_res
    simpar['l0'] = l0
    simpar['k0'] = k0
    ant_l_tx = antennas.tx.La
    simpar['ant_l_tx'] = ant_l_tx
    if getattr(cfg.mode, 'mode', 'stripmap').lower() == 'stripmap' and not hasattr(cfg.mode, 'aperture_time'):
        simpar['antenna_sine_span'] = antennas.stripmap_sine_span()

    # Complete receive antennas are translated independently of their elements.
    simpar['b_ati'], simpar['b_xti'] = b_ati, b_xti
    k0_b_ati, k0_b_xti = k0 * b_ati, k0 * b_xti

    derive_v_ground = v_ground == 'auto'
    if derive_v_ground:
        provisional_v_ground = geosar.orbit_to_vel(
            alt, ground=True, inc=inc_angle)
    else:
        provisional_v_ground = v_ground
    simpar['v_ground'] = provisional_v_ground
    provisional_t_span = nominal_raw_time_span(cfg, simpar, surface)
    ghist = GeoHistory(
        cfg,
        #tilt=txcnf.tilt,
        #tilt_b=rxcnf.tilt,
        latitude=10,
        bistatic=bistatic,
        dau=dau,
        inc_range= np.degrees(np.array([inc_angle, inc_angle+np.radians(30)])) + np.array([-3, 3]),
        inc_swth=np.degrees(inc_angle)+ np.array([-1, 1]),
        #=verbosity,
        n_la_pts=800,
        #r_planet=r_planet,
        #gm_planet=gm_planet,
        aei=None,
        #ellipsoid=ellipsoid,
        #orb_type=orb_type,
        # The extra two seconds make this conservative with respect to the
        # provisional circular-orbit ground velocity.
        t_analysis=provisional_t_span + 2.0 + calc_analysis_time(
            cfg.orbit.Horb, inc_angle, cfg.sar.f0, cfg.mode.prf, n_amb=1)

    )
    # recompute sr using ghist
    sr = ghist.sr_spl(la, 0).flatten()
    simpar['sr0'] = sr[0]
    sr0 = simpar['sr0']
    sr_near = sr[0] - wh_tol + const.c / 2 / (rg_sampling)
    simpar['sr_near'] = sr_near
    sr = sr - np.min(sr)
    sr = sr.reshape(1, sr.size)
    if derive_v_ground:
        v_ground = geosar.geohistory_ground_velocity(
            ghist, la[la.size // 2])
        info.msg(
            "GeoHistory ground velocity: %.3f m/s "
            "(provisional value: %.3f m/s)"
            % (v_ground, provisional_v_ground),
            importance=2)
    else:
        v_ground = provisional_v_ground
    simpar['v_ground'] = v_ground
    parameter_builder = factorize_raw_params if factorization is None else factorization.parameters
    simpar = parameter_builder(cfg, simpar, surface, info)
    # Get range of azimut angles to intialize Bragg model..
    # TODO
    # angular range
    az_span = (simpar["t_span"] * v_ground + surface.Ly) /gr0

    # Number of RG samples
    max_sr = np.max(sr) + wh_tol + (np.max(surface.y) + (simpar["t_span"]/2.)*v_ground)**2./(2.*sr0)
    min_sr = np.min(sr) - wh_tol
    rg_samp_orig = int(np.ceil(((max_sr - min_sr)/sr_res)*over_fs))
    rg_samp = int(drtls.optimize_fftsize(rg_samp_orig))

    # Other initializations
    proc_raw_vv = None # Just to avoid errors in the code, these will be initialized later
    proc_raw_hh = None
    # These are already complex residual fields, not large slant ranges.
    # Store the retained block/time cube in complex64; double precision is
    # required for range/phase geometry, not for its field mantissas. Keep
    # the much smaller row accumulator in complex128 so this storage
    # change does not also change scatterer summation precision/order.
    if do_hh:
        proc_raw_hh = np.zeros([num_ch, simpar["az_steps"], simpar["nblocks"], rg_samp], dtype=np.complex64)
    if do_vv:
        proc_raw_vv = np.zeros([num_ch, simpar["az_steps"], simpar["nblocks"],rg_samp], dtype=np.complex64)
    sr_surface_fct = np.zeros([simpar["az_steps"], simpar['nblocks']])
    sr_surface_fct_full = np.zeros([simpar["az_steps"]*simpar["n_pulses_b"], simpar['nblocks']])
    # sr_surface_fct[::int(surface.Ny/simpar['nblocks'])]
    NRCS_avg_vv = np.zeros(simpar["az_steps"], dtype=float)
    NRCS_avg_hh = np.zeros(simpar["az_steps"], dtype=float)
    sr_pt_history = np.zeros(simpar["az_steps"], dtype=float)
    sr_pt_y = int(surface.Ny/2)
    sr_pt_x = int(surface.Nx/2)
    pt_y_edge = int(round(0.1*(surface.Ny - 1)))
    pt_x_edge = int(round(0.1*(surface.Nx - 1)))
    point_targets = tuple(
        (target_y, target_x)
        for target_y in (pt_y_edge, sr_pt_y,
                         int(surface.Ny - 1 - pt_y_edge))
        for target_x in (pt_x_edge, sr_pt_x,
                         int(surface.Nx - 1 - pt_x_edge)))
    sr_pt_block = sr_pt_y//simpar['block_Ny']


    ## RCS MODELS
    if scat_spec_enable:
        rcs_spec = rcs.RCSKodis(inc, k0, surface.dx, surface.dy)
    # Bragg
    if scat_bragg_enable:
        tau_c = closure.grid_coherence(cfg.ocean.wind_U,surface.dx, f0)
        info.msg("Surface grid coherence time: %f s" % (tau_c))
        rndscat_p = closure.randomscat_ts(
            tau_c, (surface.Ny, surface.Nx), 1/simpar["t_step"],
            rng=bragg_positive_rng)
        rndscat_m = closure.randomscat_ts(
            tau_c, (surface.Ny, surface.Nx), 1/simpar["t_step"],
            rng=bragg_negative_rng)
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
                                          scat_bragg_d,
                                          rmss_x=surface.rmss_x,
                                          rmss_y=surface.rmss_y,
                                          az_span=az_span,
                                          use_lut=use_lut)
        else:
            raise NotImplementedError('RCS model %s for Bragg scattering not implemented' % scat_bragg_model)

    surface_area = surface.dx * surface.dy * surface.Nx * surface.Ny
    # Reference block wise range history in case we factorize this
    info.msg("Initialize reference range history",1)
    block_centres = np.arange(simpar['block_Ny']//2, surface.Ny, simpar['block_Ny'])
    for az_step in tqdm(range(simpar["az_steps"]*simpar["n_pulses_b"])):
        t_now = az_step * simpar["t_step"]/simpar["n_pulses_b"]
        az_now = simpar["az0"] + t_now * simpar["v_ground"]
        az = (surface.y - az_now).reshape((surface.Ny, 1))
        t_az = -az / simpar["v_ground"]
        sr_surface_fct_ = np.flip(ghist.sr_spl(la[0], np.flip(t_az.flatten()))).flatten() - simpar['sr0']
        sr_surface_fct_ = sr_surface_fct_[block_centres]
        block_sin_az = np.flip(ghist.t2u_spl(la[la.size//2],
            np.flip(t_az[block_centres, 0])), axis=1).flatten()
        sr_surface_fct_ = sr_surface_fct_ + factorization_current_range(
            cfg, simpar, surface, t_now, sin_inc[0, 0], block_sin_az)
        sr_surface_fct_full[az_step] = sr_surface_fct_
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    im = axs[0].imshow(sr_surface_fct_full, aspect='auto')
    plt.colorbar(im, ax=axs[0])
    axs[0].set_title('Reference range history for factorization')
    im = axs[1].imshow(sr_surface_fct_full-(sr_surface_fct_full[:,0])[:,np.newaxis], aspect='auto')
    plt.colorbar(im, ax=axs[1])
    axs[1].set_title('Residual reference range history for factorization')

    def compute_profiles():
        """Fill retained coarse profiles using complete factorization blocks.

        Ocean FFT fields and RCS power caches remain global in space. Each
        work group uses views into those fields and the existing packed random
        scatterer indices; no RNG is created inside the chunk loop. Local
        temporaries are released on return, before full-PRF reconstruction.
        """
        profiles = {'hh': proc_raw_hh, 'vv': proc_raw_vv}
        profiles = {name: field for name, field in profiles.items() if field is not None}
        nrcs = {'hh': NRCS_avg_hh, 'vv': NRCS_avg_vv}
        # A single REAL power map preserves the reference's full-grid NumPy
        # reduction order for bitwise NRCS statistics. Complex specular fields
        # stay chunk-local and are calculated only once for both polarizations.
        specular_power = np.empty((surface.Ny, surface.Nx)) if scat_spec_enable else None
        rcs_cache = {}
        last_rcs_time = -1.
        block_rows = simpar['block_Ny']
        info.msg('Computing profiles with %d factorization blocks per chunk' % blocks_per_chunk)
        for az_step in tqdm(range(simpar['az_steps'])):
            t_now = az_step * simpar['t_step']
            az_now = simpar['az0'] + t_now * simpar['v_ground']
            if az_step == 0 or not cfg.ocean.frozen_ocean:
                scene_time = factorization_scene_time(cfg, simpar, t_now)
                # The global Fourier surface is evolved ONCE per coarse time,
                # never once per spatial chunk. Independent tile FFTs would
                # produce a different ocean and duplicate substantial work.
                surface.t = scene_time
            az = (surface.y - az_now).reshape((surface.Ny, 1))
            t_az = -az / simpar['v_ground']
            sin_az_all = np.flip(
                ghist.t2u_spl(la[la.size // 2], np.flip(t_az.flatten())), axis=1).T
            az_proj_angle = np.arcsin(az / simpar['gr0'])

            update_rcs = scat_bragg_enable and (
                last_rcs_time < 0 or (not cfg.ocean.frozen_ocean
                                     and (t_now - last_rcs_time) > ocean_dt))
            if update_rcs:
                # Preserve the reference's cache cadence, including frozen
                # snapshots and coarse intervals shorter than ocean.dt. Fill
                # the cache by row groups so LUT indices/interpolation results
                # never need another full-scene working array.
                if use_hmtf:
                    surface.hMTF[np.where(surface.hMTF < -1)] = -1
                for first in range(0, simpar['nblocks'], blocks_per_chunk):
                    stop = min(first + blocks_per_chunk, simpar['nblocks'])
                    rows = slice(first * block_rows, stop * block_rows)
                    powers = rcs_bragg.rcs(az_proj_angle[rows],
                                           surface.Diffx[rows], surface.Diffy[rows])
                    powers = dict(zip(('hh', 'vv'), powers)) if pol == 'DP' else {next(iter(profiles)): powers}
                    for name, power in powers.items():
                        if use_hmtf:
                            power[0] *= (1 + surface.hMTF[rows])
                            power[1] *= (1 + surface.hMTF[rows])
                        if name not in rcs_cache:
                            # LUT advanced indexing interleaves the two Bragg
                            # directions; the direct model has a different axis
                            # order. Preserve either layout so np.sum retains
                            # the reference's reduction order and NRCS bits.
                            shape = (2, surface.Ny, surface.Nx)
                            order = np.argsort(power.strides)[::-1]
                            storage = np.empty(tuple(shape[i] for i in order), dtype=power.dtype)
                            rcs_cache[name] = storage.transpose(np.argsort(order))
                        rcs_cache[name][:, rows, :] = power
                last_rcs_time = t_now
            # Keep the same full-cache reduction and memory layout so grouping
            # work does not also change floating-point NRCS reduction order.
            bragg_nrcs = ({name: np.sum(rcs_cache[name]) / surface_area for name in profiles}
                          if scat_bragg_enable else {})

            for first in range(0, simpar['nblocks'], blocks_per_chunk):
                stop = min(first + blocks_per_chunk, simpar['nblocks'])
                row_start, row_stop = first * block_rows, stop * block_rows
                rows = slice(row_start, row_stop)
                nrows = row_stop - row_start
                sin_az = sin_az_all[rows]
                # Absolute satellite ranges remain float64 until the global
                # origin has been subtracted and the complex phase formed.
                sr_history = np.flip(
                    ghist.sr_spl(la, np.flip(t_az[rows].flatten())), axis=1).T
                sr_history -= simpar['sr0']
                sr_surface = (sr_history - cos_inc * surface.Dz[rows]
                              + surface.Dx[rows] * sin_inc + surface.Dy[rows] * sin_az)
                local_centres = np.arange(block_rows // 2, nrows, block_rows)
                reference = sr_history[local_centres, 0] + factorization_current_range(
                    cfg, simpar, surface, t_now, sin_inc[0, 0], sin_az[local_centres, 0])
                sr_surface_fct[az_step, first:stop] = reference
                sr_surface = (sr_surface.reshape(stop-first, block_rows, surface.Nx)
                              - reference[:, None, None]).reshape(nrows, surface.Nx)
                fields = {name: np.zeros((nrows, surface.Nx), dtype=complex) for name in profiles}
                if row_start <= sr_pt_y < row_stop:
                    sr_pt_history[az_step] = (sr_history[sr_pt_y-row_start, sr_pt_x]
                                              - reference[sr_pt_block-first])
                if add_point_target:
                    for target_y, target_x in point_targets:
                        if row_start <= target_y < row_stop:
                            local_y = target_y - row_start
                            target_block = target_y // block_rows
                            sr_target = sr_history[local_y, target_x] - reference[target_block-first]
                            pt_scat = 100. * np.exp(-1j * 2. * k0 * sr_target)
                            for scene in fields.values():
                                scene[local_y, target_x] = pt_scat
                            sr_surface[local_y, target_x] = sr_target
                if scat_spec_enable:
                    Esn_sp = rcs.specular_field(
                        rcs_spec, az_proj_angle[rows], sr_surface, inc, surface.Dz[rows],
                        surface.Diffx[rows], surface.Diffy[rows], surface.Diffxx[rows],
                        surface.Diffyy[rows], surface.Diffxy[rows])
                    specular_power[rows] = np.abs(Esn_sp)**2
                    for scene in fields.values():
                        scene += Esn_sp
                if scat_bragg_enable:
                    surf_phase = -(2 * k0) * sr_surface
                    if cfg.ocean.frozen_ocean:
                        cap_phase = 0.0
                    else:
                        cap_phase = (2 * k0) * simpar['t_step'] * c_b * (az_step + 1)
                        cap_phase += (2*k0)*c_b*simpar.get('scene_time_offset', 0.)
                    phase_bragg = np.empty((2, nrows, surface.Nx))
                    phase_bragg[0] = surf_phase - cap_phase
                    phase_bragg[1] = surf_phase + cap_phase
                    bragg_scats = np.empty((2, nrows, surface.Nx), dtype=complex)
                    scatter_time = simpar.get('frozen_scene_time', 0.) if cfg.ocean.frozen_ocean else scene_time
                    rndscat_m.scats(scatter_time, index=(rows, slice(None)), out=bragg_scats[0])
                    rndscat_p.scats(scatter_time, index=(rows, slice(None)), out=bragg_scats[1])
                    for name, scene in fields.items():
                        RCS_power = rcs_cache[name][:, rows, :]
                        scene += ne.evaluate('sum(sqrt(RCS_power) * exp(1j*phase_bragg) * bragg_scats, axis=0)')

                wave_dinc = ((surface.Dz[rows] * sin_inc + surface.Dx[rows] * sin_inc) / simpar['sr0']
                             if has_cross_track_baseline else None)
                tot_dinc = ((inc - inc_angle) + wave_dinc if has_cross_track_baseline else None)
                beam_pattern = antennas.field(sin_az)
                for scene in fields.values():
                    scene *= beam_pattern
                # Same complex128 per-row scatterer accumulation, then the
                # same block_Ny-row NumPy sum and complex64 storage as sar_raw.
                row_profiles = np.zeros((nrows, rg_samp), dtype=complex)
                for ch in np.arange(num_ch, dtype=int):
                    if k0_b_xti[ch] == 0:
                        chan_phase = (1. if k0_b_ati[ch] == 0
                                      else np.exp(-1j * k0_b_ati[ch] * sin_az))
                    else:
                        chan_phase_arg = (k0_b_xti[ch] * tot_dinc if k0_b_ati[ch] == 0
                                          else k0_b_ati[ch] * sin_az + k0_b_xti[ch] * tot_dinc)
                        chan_phase = ne.evaluate('cos(chan_phase_arg) - 1j * sin(chan_phase_arg)')
                    for name, scene in fields.items():
                        scene_bp = scene * chan_phase
                        row_profiles[:] = 0
                        # Possible future OPTION: prepare range-bin indices
                        # once per chunk and reuse across channels/pols. The
                        # phase-only baseline model shares their range geometry.
                        # Leave chan_profile_numba entirely unchanged for this
                        # equivalence/overhead test, including its repeated work.
                        raw.chan_profile_numba(sr_surface, scene_bp, sr_res/over_fs,
                                               min_sr, chan_sinc_vec, n_sinc_samples,
                                               sinc_ovs, row_profiles, rg_only=True)
                        profiles[name][ch, az_step, first:stop] = np.sum(
                            row_profiles.reshape(stop-first, block_rows, rg_samp), axis=1)

            # Use the same specular-then-Bragg additions as the reference.
            # HH and VV receive identical specular field AND power, while
            # their Bragg terms keep their original polarization dependence.
            if scat_spec_enable:
                spec_nrcs = np.sum(specular_power) / surface_area
                for name in profiles:
                    nrcs[name][az_step] += spec_nrcs
            for name, value in bragg_nrcs.items():
                nrcs[name][az_step] += value

    compute_profiles()
    # PROCESS REDUCED RAW DATA & SAVE (ROOT)
    # Now we need to upsample, and restore the full RCM and phase
    # sr_surface_fct
    # proc_raw_vv
    # proc_raw_hh
    aggregate = aggregate_factorized_raw if factorization is None else factorization.aggregate
    proc_raw_hh, proc_raw_vv = aggregate(proc_raw_hh,
                                                        proc_raw_vv,
                                                        sr_surface_fct, sr_surface_fct_full,
                                                        simpar, surface, cfg, info, workers=cfg.sim.nworkers)
    sr_pt_history = drtls.quadresample(
        sr_pt_history,
        np.arange(simpar["az_steps"]*simpar["n_pulses_b"])/simpar["n_pulses_b"],
        extrapolate=True)
    sr_pt_history = sr_pt_history + sr_surface_fct_full[:, sr_pt_block]
    crop_start = simpar["az_crop_start"]
    crop_stop = crop_start + simpar["output_az_steps"]
    sr_pt_history = sr_pt_history[crop_start:crop_stop]
    info.msg('Processing and saving results...')

    # Filter and decimate
    #range_filter = np.ones_like(total_raw)
    #range_filter[:, :, rg_samp/(2*2*cfg.mode.over_fs):-rg_samp/(2*2*cfg.mode.over_fs)] = 0

    #total_raw = np.fft.ifft(range_filter*np.fft.fft(total_raw))
    if do_hh:
        proc_raw_hh = proc_raw_hh[:, :, :rg_samp_orig]
    if do_vv:
        proc_raw_vv = proc_raw_vv[:, :, :rg_samp_orig]

    # Store the amplitude calibration without applying it to ideal samples.
    # The system-errors stage currently calibrates before adding NESZ noise,
    # keeping that unit conversion beside the receiver model for future changes.
    az_axis = np.linspace(-simpar["t_span"]/2*v_ground,
                          simpar["t_span"]/2*v_ground, 8193)

    # Use the same GeoHistory direction cosine and complex field as acquisition.
    cal_times = -az_axis/v_ground
    cal_sine = np.flip(ghist.t2u_spl(la[la.size//2], np.flip(cal_times)), axis=1).flatten()
    pattern = antennas.field(cal_sine)
    illuminated_area = trapezoid(np.abs(pattern)**2., az_axis)*sr_res/np.sin(inc_angle)
    if not np.isfinite(illuminated_area) or illuminated_area <= 0:
        raise ValueError("Antenna has no finite illuminated calibration area")
    cal_factor = 1./np.sqrt(illuminated_area)

    # Save RAW data (and other properties, used by 3rd party software)
    if do_hh and do_vv:
        rshp = (1,) + proc_raw_hh.shape
        total_raw = np.concatenate((proc_raw_hh.reshape(rshp),
                                    proc_raw_vv.reshape(rshp)))
        rshp = (1,) + NRCS_avg_hh.shape
        NRCS_avg = np.concatenate((NRCS_avg_hh.reshape(rshp),
                                   NRCS_avg_vv.reshape(rshp)))
    elif do_hh:
        rshp = (1,) + proc_raw_hh.shape
        total_raw = proc_raw_hh.reshape(rshp)
        rshp = (1,) + NRCS_avg_hh.shape
        NRCS_avg = NRCS_avg_hh.reshape(rshp)
    else:
        rshp = (1,) + proc_raw_vv.shape
        total_raw = proc_raw_vv.reshape(rshp)
        rshp = (1,) + NRCS_avg_vv.shape
        NRCS_avg = NRCS_avg_vv.reshape(rshp)

    full_az_steps = simpar["az_steps"] * simpar["n_pulses_b"]
    NRCS_avg = drtls.linresample(
        NRCS_avg,
        np.arange(full_az_steps) / simpar["n_pulses_b"],
        axis=1,
        extrapolate=True)
    crop_start = simpar["az_crop_start"]
    crop_stop = crop_start + simpar["output_az_steps"]
    NRCS_avg = NRCS_avg[:, crop_start:crop_stop]

    if do_vv:
        plt.figure()
        plt.imshow(np.abs(proc_raw_vv[0]),
                   origin='lower',
                   cmap='inferno_r')

        #plt.grid(True)
        #pltax = plt.gca()
        #pltax.set_xlim((-0.1, 0.1))
        #pltax.set_ylim((-0.1, 0.1))
        #plt.xlabel('$k_x$ [rad/m]')
        #plt.ylabel('$k_y$ [rad/m]')
        #plt.colorbar()
        #plt.show()
        # Create plots directory
        plot_path = os.path.dirname(output_file) + os.sep + 'raw_plots'
        if plot_save:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
        plt.savefig(os.path.join(plot_path, "raw_vv.png"))
        plt.close()

    raw_file = tpio.RawFile(output_file, 'w', total_raw.shape)
    antennas.save(raw_file.__file__)
    raw_file.__file__.raw_generator_variant = 'factorized-azimuth-chunks-v1'
    raw_file.__file__.factorization_blocks_per_chunk = int(blocks_per_chunk)
    raw_file.__file__.raw_data_state = 'ideal'
    raw_file.__file__.calibration_applied = 0
    raw_file.__file__.factorization_profile_dtype = 'complex64'
    raw_file.__file__.factorization_current_reference = int(getattr(cfg.srg, 'factorize_current', True))
    raw_file.__file__.factorization_current_vector_m_s = np.asarray(surface.current, dtype=float)
    raw_file.__file__.factorization_coarse_sampling_hz = float(1/simpar['t_step'])
    if 'scansar_guard_coarse_samples' in simpar:
        raw_file.__file__.factorization_scansar_guard_coarse_samples = int(
            simpar['scansar_guard_coarse_samples'])
        raw_file.__file__.factorization_coarse_sample_count = int(simpar['az_steps'])
        full_count = simpar['az_steps'] * simpar['n_pulses_b']
        raw_file.__file__.factorization_azimuth_padding_full_prf_samples = np.array([
            simpar['az_crop_start'],
            full_count - simpar['az_crop_start'] - simpar['output_az_steps'],
        ], dtype=np.int64)
    raw_file.__file__.factorization_current_clock = ('fixed ocean snapshot' if cfg.ocean.frozen_ocean
                                                    else 'elapsed generator time plus scene_time_offset')
    raw_file.__file__.factorization_current_scene_time_offset_s = float(simpar.get('scene_time_offset', 0.))
    raw_file.__file__.factorization_current_frozen_scene_time_s = float(simpar.get('frozen_scene_time', 0.))
    raw_file.__file__.createVariable('cal_factor', 'f8')
    raw_file.__file__['cal_factor'].long_name = 'Amplitude scale from simulator field units to calibrated raw data'
    raw_file.set('cal_factor', cal_factor)
    raw_file.set('inc_angle', np.rad2deg(inc_angle))
    raw_file.set('f0', f0)
    raw_file.set('num_ch', num_ch)
    raw_file.set('ant_L', ant_l_tx)
    metadata.write_sampling(raw_file.__file__, prf, prf)
    labels = ['hh', 'vv'] if str(pol).lower() == 'dp' else [str(pol).lower()]
    metadata.write_geometry(raw_file.__file__, cfg, labels, surface=surface)
    raw_file.set('v_ground', v_ground)
    raw_file.set('az0', simpar['output_az0'])
    raw_file.set('orbit_alt', alt)
    raw_file.set('sr0', sr_near)
    raw_file.set('rg_sampling', rg_bw*over_fs)
    raw_file.set('rg_bw', rg_bw)
    if sr_pt_history.shape[0] != total_raw.shape[2]:
        raise ValueError('sr_pt azimuth samples do not match raw az_dim')
    raw_file.set('sr_pt', sr_pt_history)
    raw_file.set('raw_data*', total_raw)
    info.msg(NRCS_avg.shape)
    info.msg(total_raw.shape)
    raw_file.set('NRCS_avg', NRCS_avg)
    raw_file.set('b_ati', b_ati)
    raw_file.set('b_xti', b_xti)
    raw_file.close()

    print(time.strftime("Finished [%Y-%m-%d %H:%M:%S]", time.localtime()))


def main(argv=None):
    """Generate one factorized raw product from CLI paths; return None.

    argv is an optional list of command-line strings. Positional arguments
    supply configuration, fresh raw output and ocean initial-condition paths;
    --blocks-per-chunk controls work grouping, not physical factorization.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('output')
    parser.add_argument('ocean')
    parser.add_argument('--reuse-ocean', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--blocks-per-chunk', type=int, default=16)
    args = parser.parse_args(argv)
    sar_raw_factorized_chunked(args.config, args.output, args.ocean,
                              args.reuse_ocean, blocks_per_chunk=args.blocks_per_chunk,
                              overwrite=args.overwrite)


if __name__ == '__main__':
    main()
