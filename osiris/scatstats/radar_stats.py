
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import numexpr as ne
from osiris import io
from osiris import utils
import osiris.utils.geometry as geosar
from osiris import constants as const

from osiris import nrcs as rcs
from osiris.surfaces import OceanSurface
from osiris import closure
import matplotlib as mpl

def surface_S(cfg_file=None, inc_deg=None, ntimes=2, t_step=10e-3):
    """ This function generates a (short) time series of surface realizations.

        :param scf_file: the full path to the configuration with all OSIRIS parameters
        :param inc_deg: the incident angle, in degree
        :param ntimes: number of time samples generated.
        :param t_step: spacing between time samples. This can be interpreted as the Pulse Repetition Interval

        :returns: a tuple with the configuration object, the surfaces, the radial velocities for each grid point,
                  and the complex scattering coefficients
    """

    cfg_file = utils.get_parFile(parfile=cfg_file)
    cfg = io.ConfigFile(cfg_file)
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
    alt = cfg.sar.alt
    f0 = cfg.sar.f0
    prf = cfg.sar.prf
    pol = cfg.sar.pol
    l0 = const.c / f0
    k0 = 2.*np.pi*f0/const.c
    if pol == 'DP':
        do_hh = True
        do_vv = True
    elif pol == 'hh':
        do_hh = True
        do_vv = False
    else:
        do_hh = False
        do_vv = True
    # OCEAN / OTHERS
    ocean_dt = cfg.ocean.dt
    surface = OceanSurface()
    compute = ['D', 'Diff', 'Diff2','V']
    if use_hmtf:
        compute.append('hMTF')
    surface.init(cfg.ocean.Lx, cfg.ocean.Ly, cfg.ocean.dx,
                 cfg.ocean.dy, cfg.ocean.cutoff_wl,
                 cfg.ocean.spec_model, cfg.ocean.spread_model,
                 np.deg2rad(cfg.ocean.wind_dir),
                 cfg.ocean.wind_fetch, cfg.ocean.wind_U,
                 cfg.ocean.current_mag,
                 np.deg2rad(cfg.ocean.current_dir),
                 cfg.ocean.swell_enable, cfg.ocean.swell_ampl,
                 np.deg2rad(cfg.ocean.swell_dir),
                 cfg.ocean.swell_wl,
                 compute, cfg.ocean.opt_res,
                 cfg.ocean.fft_max_prime,
                 choppy_enable=cfg.ocean.choppy_enable)
    # Get a surface realization calculated
    surface.t = 0
    if inc_deg is None:
        inc_deg = cfg.sar.inc_angle

    inc_angle = np.radians(inc_deg)
    sr0 = geosar.inc_to_sr(inc_angle, alt)
    gr0 = geosar.inc_to_gr(inc_angle, alt)
    gr = surface.x + gr0
    sr, inc, _ = geosar.gr_to_geo(gr, alt)
    sr -= np.min(sr)
    inc = inc.reshape(1, inc.size)
    sr = sr.reshape(1, sr.size)
    gr = gr.reshape(1, gr.size)
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)

    t_last_rcs_bragg = -1.
    last_progress = -1
    NRCS_avg_vv = np.zeros(ntimes, dtype=np.float)
    NRCS_avg_hh = np.zeros(ntimes, dtype=np.float)
    # RCS MODELS
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
        tau_c = closure.grid_coherence(cfg.ocean.wind_U,
                                       surface.dx, f0)
        rndscat_p = closure.randomscat_ts(tau_c, (surface.Ny, surface.Nx), prf)
        rndscat_m = closure.randomscat_ts(tau_c, (surface.Ny, surface.Nx), prf)
        # NOTE: This ignores slope, may be changed
        k_b = 2. * k0 * sin_inc
        c_b = sin_inc * np.sqrt(const.g/k_b + 0.072e-3 * k_b)

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
    if do_hh:
        scene_hh = np.zeros([ntimes, surface.Ny, surface.Nx], dtype=np.complex)
    if do_vv:
        scene_vv = np.zeros([ntimes, surface.Ny, surface.Nx], dtype=np.complex)

    for az_step in range(ntimes):

        # AZIMUTH & SURFACE UPDATE
        t_now = az_step * t_step
        # az_now = (t_now - t_span/2.)*v_ground
        az_now = 0
        # az = np.repeat((surface.y - az_now)[:, np.newaxis], surface.Nx, axis=1)
        az = (surface.y - az_now).reshape((surface.Ny, 1))
        surface.t = t_now


        ## COMPUTE RCS FOR EACH MODEL
        # Note: SAR processing is range independent as slant range is fixed
        sin_az = az / sr0
        az_proj_angle = np.arcsin(az / gr0)

        # Note: Projected displacements are added to slant range
        sr_surface = (sr - cos_inc*surface.Dz + az/2*sin_az +
                      surface.Dx*sin_inc + surface.Dy*sin_az)



        # Specular
        if scat_spec_enable:
            if scat_spec_mode == 'kodis':
                Esn_sp = np.sqrt(4.*np.pi)*rcs_spec.field(az_proj_angle,
                                                          sr_surface,
                                                          surface.Diffx,
                                                          surface.Diffy,
                                                          surface.Diffxx,
                                                          surface.Diffyy,
                                                          surface.Diffxy)
                if do_hh:
                    scene_hh[az_step] += Esn_sp
                if do_vv:
                    scene_vv[az_step] += Esn_sp
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
                    scene_hh[az_step] += Esn_sp
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
                    scene_vv[az_step] += Esn_sp
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
                    elif pol == 'hh':
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
            phase_bragg[0] = surf_phase - cap_phase  # + dop_phase_p
            phase_bragg[1] = surf_phase + cap_phase  # + dop_phase_m
            bragg_scats[0] = rndscat_m.scats(t_now)
            bragg_scats[1] = rndscat_p.scats(t_now)
            if do_hh:
                scene_hh[az_step] += ne.evaluate('sum(scat_bragg_hh * exp(1j*phase_bragg) * bragg_scats, axis=0)')
            if do_vv:
                scene_vv[az_step] += ne.evaluate('sum(scat_bragg_vv * exp(1j*phase_bragg) * bragg_scats, axis=0)')

    v_r = (surface.Vx * np.sin(inc) -
           surface.Vz * np.cos(inc))
    if do_hh and do_vv:
        return (cfg, surface.Dz, v_r, scene_hh, scene_vv)
    elif do_hh:
        return (cfg, surface.Dz, v_r, scene_hh)
    else:
        return (cfg, surface.Dz, v_r, scene_vv)


def view_surface(S, ml=2):
    w_h = S[1]
    v_r = S[2]
    cfg = S[0]
    v_r_mean = np.mean(v_r)
    v_r_std = np.std(v_r)
    plt.figure()
    plt.imshow(w_h, aspect='equal',cmap=mpl.cm.winter,
               extent=[0., cfg.ocean.Lx, 0, cfg.ocean.Ly], origin='lower',
               vmin=-(np.abs(w_h).max()),
               vmax=(np.abs(w_h).max()))
    plt.xlabel('Ground range [m]')
    plt.ylabel('Azimuth [m]')
    plt.title('Surface Height')
    plt.colorbar()
    sigma = (utils.smooth(np.abs(S[-1][0])**2, ml) /
             cfg.ocean.dx / cfg.ocean.dy)
    s_mean = np.mean(sigma)
    dmin = np.max([0, s_mean - 1.5 * np.std(sigma)])
    dmin = utils.db(s_mean - 2 * np.std(sigma))
    #dmin = 0
    dmax = utils.db(s_mean + 2 * np.std(sigma))
    dmin = dmax - 15
    dmax = utils.db(sigma.max())
    # utils.image(utils.db(sigma), aspect='equal', cmap=utils.sea_cmap,
    #             extent=[0., cfg.ocean.Lx, 0, cfg.ocean.Ly],
    #             xlabel='Ground range [m]', ylabel='Azimuth [m]',
    #             title='Backscattering', cbar_xlabel='dB',
    #             min=dmin, max=dmax)
    plt.figure()
    plt.imshow(utils.db(sigma), aspect='equal',cmap=mpl.cm.viridis,
               extent=[0., cfg.ocean.Lx, 0, cfg.ocean.Ly], origin='lower',
               vmin=dmin,
               vmax=dmax)
    plt.xlabel('Ground range [m]')
    plt.ylabel('Azimuth [m]')
    plt.title('Radar Scattering')
    plt.colorbar()


def v_r_stats(S, ml=1):
    """
    Taking the tuple generated by surface_S, it computes the *true* radial velocity histogram as well as the
     NRCS weighted histograms. The output illustrates the (wind) wave induced bias of the Doppler.
    :param S: tuple generated by surface_S
    :param ml: spacial averaging factor (wich currently does nothing)
    """
    # FIXME
    # implement or remove ml
    v_r = S[2]
    sigma_V = np.abs(S[-1][0])**2
    sigma_H = np.abs(S[-2][0])**2
    vmax = np.abs(v_r).max()
    h_v_r, bins = np.histogram(v_r, bins=np.int(2 * vmax * 40), density=True)
    h_v_r_Vw, kk = np.histogram(v_r, bins=np.int(2 * vmax * 40), density=True,
                                weights=sigma_V)
    h_v_r_Hw, kk = np.histogram(v_r, bins=2 * np.int(vmax * 40), density=True,
                                weights=sigma_H)
    bins_ = (bins[0:-1] + bins[1:]) / 2
    plt.figure()
    plt.plot(bins_, h_v_r, label='True')
    plt.plot(bins_, h_v_r_Vw, label='$\sigma_{VV}$ Weighted')
    plt.plot(bins_, h_v_r_Hw, label='$\sigma_{HH}$ Weighted')
    plt.xlabel("$v_r$ [m/s]")
    plt.ylabel("pdf($v_r$)")
    plt.legend()
