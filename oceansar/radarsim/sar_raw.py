"""Public SAR raw-generation dispatch and unchunked standard/reference backend."""

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
def sar_raw(cfg_file, output_file, ocean_file, reuse_ocean_file=False,
            plot_save=True, bistatic=False, dau=0, *, factorization=None):
    """Generate ideal raw data using the backend selected by the configuration.

    Parameters
    ----------
    cfg_file : str or path-like
        Simulator configuration. srg.factorize=False selects the standard
        full-PRF generator. True selects spatially chunked factorization for
        Bragg, Kodis and point targets. Factorized KA spa/fa specular modes
        retain the unchunked backend until their chunked support is available.
        srg.factorize_blocks_per_chunk is a positive integer (default 16),
        controlling workspace grouping without changing physical blocks,
        sampling, guard counts or interpolation.
    output_file : str or path-like
        Destination RawFile, replaced on a successful rerun as in the original
        interface. Samples are ideal and uncalibrated, with axes
        (polarization, channel, azimuth, range). cal_factor, sampling, geometry,
        baselines and point-target range history are stored alongside them.
    ocean_file : str or path-like
        Ocean Fourier initial conditions to create or load.
    reuse_ocean_file : bool, optional
        Reuse an initialized ocean state when possible.
    plot_save : bool, optional
        Forward diagnostic plotting selection to the backend. The existing
        backends still save some diagnostic images when this is False.
    bistatic : bool, optional
        Enable bistatic GeoHistory geometry. Scattering/antenna models retain
        their existing quasi-monostatic approximations.
    dau : float, optional
        Along-track spacecraft displacement in metres, passed to GeoHistory.
    factorization : object, optional
        Experimental parameters/aggregate strategy, forwarded unchanged to
        the selected backend, including any physical scene-clock offsets.

    Returns
    -------
    None
        Write ideal raw samples and metadata. System errors, focusing and
        temporary-product cleanup remain separate orchestration steps.

    Notes
    -----
    Notebook and simulator callers use this function for either sampling mode.
    Its memory monitor covers dispatch and the selected backend in ONE worker;
    nested backend guards do not create another process. The memory report
    keeps the established memory_sar_raw filename. Specular HH/VV share one
    field; Bragg retains polarization dependence.
    """
    cfg = tpio.ConfigFile(cfg_file)
    chunked = (cfg.srg.factorize and
               (not cfg.srg.scat_spec_enable or cfg.srg.scat_spec_mode == 'kodis'))
    if chunked:
        from oceansar.radarsim.sar_raw_factorized_chunked import sar_raw_factorized_chunked
        return sar_raw_factorized_chunked(
            cfg_file, output_file, ocean_file, reuse_ocean_file,
            plot_save=plot_save, bistatic=bistatic, dau=dau,
            blocks_per_chunk=getattr(cfg.srg, 'factorize_blocks_per_chunk', 16),
            factorization=factorization, overwrite=True)
    if cfg.srg.factorize:
        print('Using unchunked factorization for specular mode %s' % cfg.srg.scat_spec_mode)
    return sar_raw_standard(cfg_file, output_file, ocean_file, reuse_ocean_file,
                            plot_save=plot_save, bistatic=bistatic, dau=dau,
                            factorization=factorization)


@memory_limited
def sar_raw_standard(cfg_file, output_file, ocean_file, reuse_ocean_file=False,
            plot_save=True, bistatic=False, dau=0, *, factorization=None):
    """Generate ideal SAR raw samples using the unchunked standard/reference path.

    Parameters
    ----------
    cfg_file : str or path-like
        Simulator configuration defining the orbit, radar, acquisition mode,
        ocean surface and scattering model.
    output_file : str or path-like
        Destination RawFile. Complex samples are stored as raw_data_r/raw_data_i
        with axes (polarization, channel, azimuth, range), in simulator field
        units. The scalar cal_factor is an amplitude multiplier; it is saved
        without applying it. Geometry, baselines and reference range history
        are also written.
    ocean_file : str or path-like
        Ocean initial-condition file to load or create, containing native
        complex Fourier coefficients and scalar surface metadata.
    reuse_ocean_file : bool, optional
        Load ocean_file when possible instead of generating a new realization.
    plot_save : bool, optional
        Create the raw_plots diagnostic directory beside output_file. The
        current implementation still writes diagnostic images when False.
    bistatic : bool, optional
        Enable bistatic geometry in the orbit history. The scattering and
        antenna models retain their quasi-monostatic approximation.
    dau : float, optional
        Along-track spacecraft displacement in metres, passed to the geometry
        model. Default is zero.
    factorization : object, optional
        Experimental factorization strategy with ``parameters`` and
        ``aggregate`` methods matching factorize_raw_params and
        aggregate_factorized_raw. None uses the production implementation.
        The strategy may supply scene_time_offset (seconds) and
        frozen_scene_time (seconds) in its parameter dictionary to align
        physical ocean/scatterer times across different internal guards.

    Returns
    -------
    None
        Products are written to disk; no raw array is returned.

    Notes
    -----
    Public sar_raw selects this backend for full-PRF generation and for the
    factorized KA specular modes whose chunked implementation is not available.
    The original factorized path is also retained as the independent numerical
    reference for chunk-equivalence checks. Call sar_raw for normal simulation.
    The output is marked raw_data_state='ideal' and calibration_applied=0.
    Call sar_system_errors separately to apply cal_factor and add NESZ noise
    before focusing. A configured sim.seed controls the random streams.
    Specular scattering is represented by one polarization-independent field
    shared by HH/VV. Kodis is scalar; KA modes use one VV evaluation for both.
    Bragg scattering retains its polarization dependence.
    A positive sim.memory_limit_gib runs this stage in an externally monitored
    worker. Its arrays are released when the call returns; a budget violation
    raises SimulationMemoryError in the caller without killing its notebook.
    """


    print('-------------------------------------------------------------------')
    print(time.strftime("- OCEANSAR SAR RAW GENERATOR: %Y-%m-%d %H:%M:%S", time.localtime()))
    print('- Copyright (c) Gerard Marull Paretas, Paco Lopez Dekker')
    print('-------------------------------------------------------------------')

    ## CONFIGURATION FILE
    # Note: variables are 'copied' to reduce code verbosity
    cfg = resolve_acquisition(tpio.ConfigFile(cfg_file))
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
    scat_spec_mode = cfg.srg.scat_spec_mode
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
    if scat_spec_enable:
        compute = ['D', 'Diff', 'Diff2']
    else:
        compute = ['D', 'Diff']

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
    if cfg.srg.factorize:
        # These are already complex residual fields, not large slant ranges.
        # Store the retained block/time cube in complex64; double precision is
        # required for range/phase geometry, not for its field mantissas. Keep
        # the much smaller row accumulator in complex128 so this storage
        # change does not also change scatterer summation precision/order.
        if do_hh:
            proc_raw_hh = np.zeros([num_ch, simpar["az_steps"], simpar["nblocks"], rg_samp], dtype=np.complex64)
            proc_raw_hh_ = np.zeros([surface.Ny,rg_samp], dtype=complex)
        if do_vv:
            proc_raw_vv = np.zeros([num_ch, simpar["az_steps"], simpar["nblocks"],rg_samp], dtype=np.complex64)
            proc_raw_vv_ = np.zeros([surface.Ny,rg_samp], dtype=complex)
        sr_surface_fct = np.zeros([simpar["az_steps"], simpar['nblocks']])
        sr_surface_fct_full = np.zeros([simpar["az_steps"]*simpar["n_pulses_b"], simpar['nblocks']])
        # sr_surface_fct[::int(surface.Ny/simpar['nblocks'])]
    else:
        if do_hh:
            proc_raw_hh = np.zeros([num_ch, simpar["az_steps"], rg_samp], dtype=complex)
            proc_raw_hh_ = np.zeros([rg_samp], dtype=complex)
        if do_vv:
            proc_raw_vv = np.zeros([num_ch, simpar["az_steps"], rg_samp], dtype=complex)
            proc_raw_vv_ = np.zeros([rg_samp], dtype=complex)
    t_last_rcs_bragg = -1.
    last_progress = -1
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
    if cfg.srg.factorize:
        sr_pt_block = sr_pt_y//simpar['block_Ny']


    ## RCS MODELS
    # Specular
    if scat_spec_enable:
        if scat_spec_mode == 'kodis':
            rcs_spec = rcs.RCSKodis(inc, k0, surface.dx, surface.dy)
        elif scat_spec_mode == 'fa' or scat_spec_mode == 'spa':
            rcs_spec = rcs.RCSKA(scat_spec_mode, k0, surface.x, surface.y,
                                 surface.dx, surface.dy)
        else:
            raise NotImplementedError('RCS mode %s for specular scattering not implemented' % scat_spec_mode)

    # Bragg
    if scat_bragg_enable:
        phase_bragg = np.zeros([2, surface.Ny, surface.Nx])
        bragg_scats = np.zeros([2, surface.Ny, surface.Nx], dtype=complex)
        # dop_phase_p = np.random.uniform(0., 2.*np.pi, size=[surface.Ny, surface.Nx])
        # dop_phase_m = np.random.uniform(0., 2.*np.pi, size=[surface.Ny, surface.Nx])
        tau_c = closure.grid_coherence(cfg.ocean.wind_U,surface.dx, f0)
        info.msg("Surface grid coherence time: %f s" % (tau_c))
        rndscat_p = closure.randomscat_ts(
            tau_c, (surface.Ny, surface.Nx), 1/simpar["t_step"],
            rng=bragg_positive_rng)
        rndscat_m = closure.randomscat_ts(
            tau_c, (surface.Ny, surface.Nx), 1/simpar["t_step"],
            rng=bragg_negative_rng)
        if cfg.ocean.frozen_ocean:
            frozen_time = simpar.get('frozen_scene_time', 0.)
            rndscat_m.scats(frozen_time, out=bragg_scats[0])
            rndscat_p.scats(frozen_time, out=bragg_scats[1])
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
    if cfg.srg.factorize:
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

    ###################
    # SIMULATION LOOP #
    ###################

    info.msg('Computing profiles...')

    for az_step in tqdm(range(simpar["az_steps"])):

        ## AZIMUTH & SURFACE UPDATE
        t_now = az_step * simpar["t_step"]
        az_now = simpar["az0"] + t_now * simpar["v_ground"]
        # az = np.repeat((surface.y - az_now)[:, np.newaxis], surface.Nx, axis=1)
        az = (surface.y - az_now).reshape((surface.Ny, 1))
        # Materialize the initial snapshot even when freezing the ocean:
        # init/load only allocate displacement, slope and hMTF arrays with
        # np.empty. Freezing skips subsequent evolution, not initialization.
        if az_step == 0 or not cfg.ocean.frozen_ocean:
            # Guards extend the numerical interval, not the physical scene
            # epoch. An experimental strategy can keep the ocean clock tied
            # to burst time so changing guards does not change the retained
            # scene. Geometry and restored histories still use az_now above.
            scene_time = factorization_scene_time(cfg, simpar, t_now)
            surface.t = scene_time

        ## COMPUTE RCS FOR EACH MODEL
        # Note: SAR processing is range independent as slant range is fixed
        t_az = -az / simpar["v_ground"]
        # Approximate the azimuth geometry with the scene-center look angle;
        # its range dependence is expected to be negligible over the swath.
        sin_az = np.flip(
            ghist.t2u_spl(la[la.size // 2], np.flip(t_az.flatten())),
            axis=1).T
        az_proj_angle = np.arcsin(az / simpar["gr0"])

        # Note: Projected displacements are added to slant range
        sr_history = np.flip(
            ghist.sr_spl(la, np.flip(t_az.flatten())), axis=1).T
        # GeoHistory returns float64 absolute ranges. Keep that precision
        # through subtraction of the global origin and carrier-phase formation;
        # the complex64 storage decision is made only on completed profiles.
        sr_history -= simpar["sr0"]
        sr_surface = (sr_history - cos_inc*surface.Dz
                         + surface.Dx*sin_inc + surface.Dy*sin_az)
        if cfg.srg.factorize:
            # Remove both satellite range history and known current drift.
            # Leaving U*t in Dx/Dy while removing only satellite motion shifts
            # the residual ocean band towards coarse-grid Nyquist, potentially
            # aliasing it before interpolation. The identical projected U*t
            # term is restored in sr_surface_fct_full, for phase AND range
            # delay. Use the ocean clock (fixed for frozen snapshots) and the
            # actual block direction, not an independent constant-tone clock.
            sr_surface_fct_ = sr_history[block_centres, 0] + factorization_current_range(
                cfg, simpar, surface, t_now, sin_inc[0, 0], sin_az[block_centres, 0])
            sr_surface_fct[az_step] = sr_surface_fct_
            sr_surface = sr_surface.reshape([simpar['nblocks'], simpar['block_Ny'],surface.Nx])
            sr_surface = sr_surface - sr_surface_fct_[:,np.newaxis,np.newaxis]
            sr_surface = sr_surface.reshape([surface.Ny,surface.Nx])
        # Incidence corrections are used only by cross-track channel phases.
        # Check ALL resolved channel positions, using exact zero so even tiny
        # nonzero baselines retain their correction. None marks the unused
        # case without allocating either of these full-scene arrays.
        wave_dinc = ((surface.Dz * sin_inc + surface.Dx * sin_inc) / simpar["sr0"]
                     if has_cross_track_baseline else None)
        if az_step == 0:
            if do_hh:
                scene_hh = np.zeros([int(surface.Ny), int(surface.Nx)], dtype=complex)
            if do_vv:
                scene_vv = np.zeros([int(surface.Ny), int(surface.Nx)], dtype=complex)
        else:
            if do_hh:
                scene_hh[:,:] = 0 
            if do_vv:
                scene_vv[:,:] = 0 
        # Point target history
        # We keep it even if we don't add it, as it gives a reference to check
        # the processor
        if cfg.srg.factorize:
            sr_pt = sr_history[sr_pt_y, sr_pt_x] - sr_surface_fct[az_step, sr_pt_block]
        else:
            sr_pt = sr_history[sr_pt_y, sr_pt_x]
        sr_pt_history[az_step] = sr_pt
        # Point target
        if add_point_target:
            for target_y, target_x in point_targets:
                if cfg.srg.factorize:
                    target_block = target_y//simpar['block_Ny']
                    sr_target = (sr_history[target_y, target_x]
                                 - sr_surface_fct[az_step, target_block])
                else:
                    sr_target = sr_history[target_y, target_x]
                pt_scat = 100.*np.exp(-1j*2.*k0*sr_target)
                if do_hh:
                    scene_hh[target_y, target_x] = pt_scat
                if do_vv:
                    scene_vv[target_y, target_x] = pt_scat
                sr_surface[target_y, target_x] = sr_target

        # Specular: evaluate one scalar field and reuse it for HH/VV.
        if scat_spec_enable:
            Esn_sp = rcs.specular_field(
                rcs_spec, az_proj_angle, sr_surface, inc, surface.Dz,
                surface.Diffx, surface.Diffy, surface.Diffxx,
                surface.Diffyy, surface.Diffxy)
            spec_nrcs = np.sum(np.abs(Esn_sp)**2) / surface_area
            if do_hh:
                scene_hh += Esn_sp
                NRCS_avg_hh[az_step] += spec_nrcs
            if do_vv:
                scene_vv += Esn_sp
                NRCS_avg_vv[az_step] += spec_nrcs

        # Bragg
        if scat_bragg_enable:
            update_bragg_rcs = (
                t_last_rcs_bragg < 0
                or (
                    not cfg.ocean.frozen_ocean
                    and (t_now - t_last_rcs_bragg) > ocean_dt
                )
            )
            if update_bragg_rcs:

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
                NRCS_bragg_hh_instant_avg = np.sum(RCS_bragg_hh) / surface_area
                NRCS_avg_hh[az_step] += NRCS_bragg_hh_instant_avg
            if do_vv:
                NRCS_bragg_vv_instant_avg = np.sum(RCS_bragg_vv) / surface_area
                NRCS_avg_vv[az_step] += NRCS_bragg_vv_instant_avg


            # Doppler phases (Note: Bragg radial velocity taken constant!)
            surf_phase = - (2 * k0) * sr_surface
            if cfg.ocean.frozen_ocean:
                cap_phase = 0.0
            else:
                cap_phase = (
                    (2 * k0) * simpar["t_step"] * c_b * (az_step + 1)
                )
                cap_phase += (2*k0)*c_b*simpar.get('scene_time_offset', 0.)
            phase_bragg[0] = surf_phase - cap_phase # + dop_phase_p
            phase_bragg[1] = surf_phase + cap_phase # + dop_phase_m
            if not cfg.ocean.frozen_ocean:
                rndscat_m.scats(scene_time, out=bragg_scats[0])
                rndscat_p.scats(scene_time, out=bragg_scats[1])
            # Fuse power-to-amplitude conversion into the NumExpr reduction:
            # sqrt is evaluated in chunks, avoiding another full (2, Ny, Nx)
            # amplitude array per polarization. Retain RCS power for cached
            # scattering updates and NRCS statistics; phase remains float64.
            if do_hh:
                scene_hh += ne.evaluate('sum(sqrt(RCS_bragg_hh) * exp(1j*phase_bragg) * bragg_scats, axis=0)')
            if do_vv:
                scene_vv += ne.evaluate('sum(sqrt(RCS_bragg_vv) * exp(1j*phase_bragg) * bragg_scats, axis=0)')

        # ANTENNA PATTERN
        # FIXME: this assume co-located Tx and Rx, so it will not work for true bistatic configurations

        beam_pattern = antennas.field(sin_az)
        # GENERATE CHANEL PROFILES
        if cfg.srg.factorize:
            sr_surface_ = sr_surface
        else:
            sr_surface_ = sr_surface.flatten()
        tot_dinc = ((inc - inc_angle) + wave_dinc
                    if has_cross_track_baseline else None)
        if do_hh:
            scene_hh *= beam_pattern
        if do_vv:
            scene_vv *= beam_pattern
        for ch in np.arange(num_ch, dtype=int):
            # Deliberately use a narrowband, phase-only baseline model. The
            # corresponding sub-resolution range delay is omitted to control
            # profile-generation cost. Revisit this approximation for large
            # channel separations or wide bandwidths.
            if k0_b_xti[ch] == 0:
                if k0_b_ati[ch] == 0:
                    chan_phase = 1.
                else:
                    # Keep the compact azimuth-column shape. Adding
                    # 0 * tot_dinc would broadcast this over the full surface.
                    chan_phase = np.exp(-1j * k0_b_ati[ch] * sin_az)
            else:
                if k0_b_ati[ch] == 0:
                    chan_phase_arg = k0_b_xti[ch] * tot_dinc
                else:
                    chan_phase_arg = (k0_b_ati[ch] * sin_az +
                                      k0_b_xti[ch] * tot_dinc)
                chan_phase = ne.evaluate(
                    "cos(chan_phase_arg) - 1j * sin(chan_phase_arg)")
            if do_hh:
                # scene_bp = scene_hh * beam_pattern
                # # Add channel phase & compute profile
                # scene_bp *= np.exp(-1j * k0 * b_ati[ch] * sin_az)
                # # Add cross-track phase
                scene_bp = scene_hh * chan_phase
                if cfg.srg.factorize:
                    scene_bp_ = scene_bp
                else:
                    scene_bp_ = scene_bp.flatten()
                proc_raw_hh_[:] = 0
                #info.msg("Here 1")
                raw.chan_profile_numba(sr_surface_,
                                       scene_bp_,
                                       sr_res/(over_fs),
                                       min_sr,
                                       chan_sinc_vec,
                                       n_sinc_samples, sinc_ovs,
                                       proc_raw_hh_,
                                       rg_only=cfg.srg.factorize)
                #info.msg("Here 2")
                if cfg.srg.factorize:
                    proc_raw_hh[ch][az_step] = np.sum(proc_raw_hh_.reshape([simpar['nblocks'], simpar['block_Ny'], rg_samp]), axis=1)
                else:
                    proc_raw_hh[ch][az_step] = proc_raw_hh_
            if do_vv:
                # scene_bp = scene_vv * beam_pattern
                # # Add channel phase & compute profile
                # scene_bp *= np.exp(-1j * k0 * b_ati[ch] * sin_az)
                # # Add cross-track phase
                # scene_bp *= np.exp(-1j * k0 * b_xti[ch] * tot_dinc)
                scene_bp = scene_vv * chan_phase
                if cfg.srg.factorize:
                    scene_bp_ = scene_bp
                else:
                    scene_bp_ = scene_bp.flatten()     
                proc_raw_vv_[:] = 0           
                raw.chan_profile_numba(sr_surface_,
                                       scene_bp_,
                                       sr_res/(over_fs),
                                       min_sr,
                                       chan_sinc_vec,
                                       n_sinc_samples, sinc_ovs,
                                       proc_raw_vv_,
                                       rg_only=cfg.srg.factorize)
                if cfg.srg.factorize:
                     proc_raw_vv[ch][az_step] = np.sum(proc_raw_vv_.reshape([simpar['nblocks'], simpar['block_Ny'], rg_samp]), axis=1)
                else:
                    proc_raw_vv[ch][az_step] = proc_raw_vv_

        # # SHOW PROGRESS (%)
        # current_progress = int((100*az_step)/az_steps)
        # if current_progress != last_progress:
        #     last_progress = current_progress
        #     print('SP, %d' % current_progress)

    # PROCESS REDUCED RAW DATA & SAVE (ROOT)
    if cfg.srg.factorize:
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

    if cfg.srg.factorize:
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
    raw_file.__file__.raw_data_state = 'ideal'
    raw_file.__file__.calibration_applied = 0
    if cfg.srg.factorize:
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
    if cfg.srg.factorize:
        info.msg(NRCS_avg.shape)
        info.msg(total_raw.shape)
    raw_file.set('NRCS_avg', NRCS_avg)
    raw_file.set('b_ati', b_ati)
    raw_file.set('b_xti', b_xti)
    raw_file.close()

    print(time.strftime("Finished [%Y-%m-%d %H:%M:%S]", time.localtime()))
