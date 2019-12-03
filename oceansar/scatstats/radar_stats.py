
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import numexpr as ne
from oceansar import ocs_io
from oceansar import utils
import oceansar.utils.geometry as geosar
from oceansar import constants as const

from oceansar import nrcs as rcs
from oceansar.surfaces import OceanSurface
from oceansar import closure
import matplotlib as mpl
from collections import namedtuple


GMF = namedtuple('GMF', ['U10', 'azimuth', 'incident', 'NRCS_hh', 'NRCS_vv', 'v_r_whh', 'v_r_wvv',
                         'v_ATI_hh', 'v_ATI_vv',
                         'sigma_v_r', 'sigma_v_r_whh', 'sigma_v_r_wvv',
                         'surfcoh_ati_hh', 'surfcoh_ati_vv'])


class RadarSurface():

    def __init__(self, cfg_file=None, ntimes=2, t_step=10e-3, pol='DP', winddir=0, U10=None,
                 po_model=None):
        """ This function generates a (short) time series of surface realizations.

                :param scf_file: the full path to the configuration with all OCEANSAR parameters
                :param ntimes: number of time samples generated.
                :param t_step: spacing between time samples. This can be interpreted as the Pulse Repetition Interval
                :param wind:dir: to force wind direction
                :param U10: to force wind force
                :param po_model: one of None, spa (stationary phase approximation, or facet approach)

                :returns: a tuple with the configuration object, the surfaces, the radial velocities for each grid point,
                      and the complex scattering coefficients
        """
        cfg_file = utils.get_parFile(parfile=cfg_file)
        cfg = ocs_io.ConfigFile(cfg_file)
        self.cfg = cfg
        self.use_hmtf = cfg.srg.use_hmtf
        self.scat_spec_enable = cfg.srg.scat_spec_enable
        if po_model is None:
            self.scat_spec_mode = cfg.srg.scat_spec_mode
        else:
            self.scat_spec_mode = po_model
        self.scat_bragg_enable = cfg.srg.scat_bragg_enable
        self.scat_bragg_model = cfg.srg.scat_bragg_model
        self.scat_bragg_d = cfg.srg.scat_bragg_d
        self.scat_bragg_spec = cfg.srg.scat_bragg_spec
        self.scat_bragg_spread = cfg.srg.scat_bragg_spread

        # SAR
        try:
            radcfg = cfg.sar
        except AttributeError:
            radcfg = cfg.radar
        self.inc_angle = np.deg2rad(radcfg.inc_angle)
        self.alt = radcfg.alt
        self.f0 = radcfg.f0
        self.prf = radcfg.prf
        if pol is None:
            self.pol = radcfg.pol
        else:
            self.pol = pol

        l0 = const.c / self.f0
        k0 = 2. * np.pi * self.f0 / const.c
        if self.pol == 'DP':
            self.do_hh = True
            self.do_vv = True
        elif self.pol == 'hh':
            self.do_hh = True
            self.do_vv = False
        else:
            self.do_hh = False
            self.do_vv = True
        # OCEAN / OTHERS
        ocean_dt = cfg.ocean.dt
        self.surface = OceanSurface()
        compute = ['D', 'Diff', 'Diff2', 'V']
        print("Initializating surface")
        if winddir is not None:
            self.wind_dir = np.radians(winddir)
        else:
            self.wind_dir = np.deg2rad(cfg.ocean.wind_dir)
        if U10 is None:
            self.wind_u = cfg.ocean.wind_U
        else:
            self.wind_u = U10
        if self.use_hmtf:
            compute.append('hMTF')
        self.surface.init(cfg.ocean.Lx, cfg.ocean.Ly, cfg.ocean.dx,
                          cfg.ocean.dy, cfg.ocean.cutoff_wl,
                          cfg.ocean.spec_model, cfg.ocean.spread_model,
                          self.wind_dir,
                          cfg.ocean.wind_fetch, self.wind_u,
                          cfg.ocean.current_mag,
                          np.deg2rad(cfg.ocean.current_dir),
                          0, 0, 0, 0, 0, False,
                          cfg.ocean.swell_enable, cfg.ocean.swell_ampl,
                          np.deg2rad(cfg.ocean.swell_dir),
                          cfg.ocean.swell_wl,
                          compute, cfg.ocean.opt_res,
                          cfg.ocean.fft_max_prime,
                          choppy_enable=cfg.ocean.choppy_enable)
        # Get a surface realization calculated
        print("Computing surface realizations")
        self.surface.t = 0
        self.ntimes = ntimes
        self.diffx = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.diffy = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.diffxx = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.diffxy = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.diffyy = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.dx = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.dy = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.dz = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
        self.diffx[0, :, :] = self.surface.Diffx
        self.diffy[0, :, :] = self.surface.Diffy
        self.diffxx[0, :, :] = self.surface.Diffxx
        self.diffyy[0, :, :] = self.surface.Diffyy
        self.diffxy[0, :, :] = self.surface.Diffxy
        self.dx[0, :, :] = self.surface.Dx
        self.dy[0, :, :] = self.surface.Dy
        self.dz[0, :, :] = self.surface.Dz
        if self.use_hmtf:
            self.h_mtf = np.zeros((ntimes, self.surface.Ny, self.surface.Nx))
            self.h_mtf[0, :, :] = self.surface.hMTF
        self.t_step = t_step
        for az_step in range(1, ntimes):
            t_now = az_step * t_step
            self.surface.t = t_now
            self.diffx[az_step, :, :] = self.surface.Diffx
            self.diffy[az_step, :, :] = self.surface.Diffy
            self.diffxx[az_step, :, :] = self.surface.Diffxx
            self.diffyy[az_step, :, :] = self.surface.Diffyy
            self.diffxy[az_step, :, :] = self.surface.Diffxy
            self.dx[az_step, :, :] = self.surface.Dx
            self.dy[az_step, :, :] = self.surface.Dy
            self.dz[az_step, :, :] = self.surface.Dz
            if self.use_hmtf:
                self.h_mtf[az_step, :, :] = self.surface.hMTF
        self.inc_is_set = False

    def set_inc(self, inc_deg=25):
        """
        Sets the incicent angle and initializes rcs calculations
        :param inc_deg: 
        :return: 
        """
        cfg = self.cfg
        use_hmtf = cfg.srg.use_hmtf
        scat_spec_enable = self.scat_spec_enable
        scat_spec_mode = self.scat_spec_mode
        scat_bragg_enable = self.scat_bragg_enable
        scat_bragg_model = self.scat_bragg_model
        scat_bragg_d = self.scat_bragg_d
        scat_bragg_spec = self.scat_bragg_spec
        scat_bragg_spread = self.scat_bragg_spread
        self.inc_is_set = True
        self.inc_angle = np.radians(inc_deg)
        inc = np.array([self.inc_angle]).reshape((1, 1))
        k0 = 2. * np.pi * self.f0 / const.c
        pol = self.pol
        if scat_bragg_model == 'romeiser97' and scat_bragg_enable:
            current_dir = np.deg2rad(cfg.ocean.current_dir)
            current_vec = (cfg.ocean.current_mag *
                           np.array([np.cos(current_dir),
                                     np.sin(current_dir)]))
            u_dir = self.wind_dir
            u_vec = (self.wind_u *
                     np.array([np.cos(u_dir), np.sin(u_dir)]))
            u_eff_vec = u_vec - current_vec

            self.rcs_bragg = rcs.RCSRomeiser97(k0, inc, pol,
                                               self.surface.dx, self.surface.dy,
                                               linalg.norm(u_eff_vec),
                                               np.arctan2(u_eff_vec[1],
                                                          u_eff_vec[0]),
                                               self.surface.wind_fetch,
                                               scat_bragg_spec, scat_bragg_spread,
                                               scat_bragg_d)
        elif scat_bragg_enable:
            raise NotImplementedError('RCS model %s for Bragg scattering not implemented' % scat_bragg_model)

    def surface_S(self, az_deg=0):
        """ This function generates a (short) time series of surface realizations.
    
            :param az_deg: azimuth angle, in degree
            :param ntimes: number of time samples generated.
            :param t_step: spacing between time samples. This can be interpreted as the Pulse Repetition Interval
    
            :returns: a tuple with the configuration object, the surfaces, the radial velocities for each grid point,
                      and the complex scattering coefficients
        """
        if not self.inc_is_set:
            print("Set the incident angle first")
            return

        cfg = self.cfg
        use_hmtf = self.use_hmtf
        scat_spec_enable = self.scat_spec_enable
        scat_spec_mode = self.scat_spec_mode
        scat_bragg_enable = self.scat_bragg_enable
        scat_bragg_model = self.scat_bragg_model
        scat_bragg_d = self.scat_bragg_d
        scat_bragg_spec = self.scat_bragg_spec
        scat_bragg_spread = self.scat_bragg_spread

        # SAR
        try:
            radcfg = cfg.sar
        except AttributeError:
            radcfg = cfg.radar
        alt = radcfg.alt
        f0 = radcfg.f0
        prf = radcfg.prf

        pol = self.pol
        l0 = const.c / f0
        k0 = 2.*np.pi*self.f0/const.c

        do_hh = self.do_hh
        do_vv = self.do_vv
        # OCEAN / OTHERS
        ocean_dt = cfg.ocean.dt

        # Get a surface realization calculated
        # self.surface.t = 0


        inc_angle = self.inc_angle
        sr0 = geosar.inc_to_sr(inc_angle, alt)
        gr0 = geosar.inc_to_gr(inc_angle, alt)
        gr = self.surface.x + gr0
        sr, inc, _ = geosar.gr_to_geo(gr, alt)
        sr -= np.min(sr)
        inc = inc.reshape(1, inc.size)
        sr = sr.reshape(1, sr.size)
        gr = gr.reshape(1, gr.size)
        sin_inc = np.sin(inc)
        cos_inc = np.cos(inc)
        az_rad = np.radians(az_deg)
        cos_az = np.cos(az_rad)
        sin_az = np.sin(az_rad)

        t_last_rcs_bragg = -1.
        last_progress = -1
        ntimes = self.ntimes
        NRCS_avg_vv = np.zeros(ntimes, dtype=np.float)
        NRCS_avg_hh = np.zeros(ntimes, dtype=np.float)
        # RCS MODELS
        # Specular
        if scat_spec_enable:
            if scat_spec_mode == 'kodis':
                rcs_spec = rcs.RCSKodis(inc, k0, self.surface.dx, self.surface.dy)
            elif scat_spec_mode == 'fa' or scat_spec_mode == 'spa':
                spec_ph0 = np.random.uniform(0., 2.*np.pi,
                                             size=[self.surface.Ny, self.surface.Nx])
                rcs_spec = rcs.RCSKA(scat_spec_mode, k0, self.surface.x, self.surface.y,
                                     self.surface.dx, self.surface.dy)
            else:
                raise NotImplementedError('RCS mode %s for specular scattering not implemented' % scat_spec_mode)

        # Bragg
        if scat_bragg_enable:
            phase_bragg = np.zeros([2, self.surface.Ny, self.surface.Nx])
            bragg_scats = np.zeros([2, self.surface.Ny,self.surface.Nx], dtype=np.complex)
            tau_c = closure.grid_coherence(cfg.ocean.wind_U,
                                           self.surface.dx, f0)
            rndscat_p = closure.randomscat_ts(tau_c, (self.surface.Ny, self.surface.Nx), prf)
            rndscat_m = closure.randomscat_ts(tau_c, (self.surface.Ny, self.surface.Nx), prf)
            # NOTE: This ignores slope, may be changed
            k_b = 2. * k0 * sin_inc
            c_b = sin_inc * np.sqrt(const.g/k_b + 0.072e-3 * k_b)

        surface_area = self.surface.dx * self.surface.dy * self.surface.Nx * self.surface.Ny
        if do_hh:
            scene_hh = np.zeros([ntimes, self.surface.Ny, self.surface.Nx], dtype=np.complex)
        if do_vv:
            scene_vv = np.zeros([ntimes, self.surface.Ny, self.surface.Nx], dtype=np.complex)

        for az_step in range(ntimes):

            # AZIMUTH & SURFACE UPDATE
            t_now = az_step * self.t_step

            ## COMPUTE RCS FOR EACH MODEL
            # Note: SAR processing is range independent as slant range is fixed
            # sin_az = az / sr0
            # az_proj_angle = np.arcsin(az / gr0)

            # Note: Projected displacements are added to slant range
            sr_surface = (sr - cos_inc*self.dz[az_step] +
                          sin_inc * (self.dx[az_step] * cos_az + self.dy[az_step] * sin_az))



            # Specular
            if scat_spec_enable:
                if scat_spec_mode == 'kodis':
                    Esn_sp = np.sqrt(4.*np.pi)*rcs_spec.field(az_rad,
                                                              sr_surface,
                                                              self.diffx[az_step],
                                                              self.diffy[az_step],
                                                              self.diffxx[az_step],
                                                              self.diffyy[az_step],
                                                              self.diffxy[az_step])
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
                                                 az_rad, az_rad + np.pi,
                                                 self.dz[az_step],
                                                 self.diffx[az_step], self.diffy[az_step],
                                                 self.diffxx[az_step],
                                                 self.diffyy[az_step],
                                                 self.diffxy[az_step]))
                        scene_hh[az_step] += Esn_sp
                    if do_vv:
                        pol_tmp = 'vv'
                        Esn_sp = (np.exp(-1j*(2.*k0*sr_surface)) * (4.*np.pi)**1.5 *
                                  rcs_spec.field(1, 1, pol_tmp[0], pol_tmp[1],
                                                 inc, inc,
                                                 az_rad, az_rad + np.pi,
                                                 self.dz[az_step],
                                                 self.diffx[az_step], self.diffy[az_step],
                                                 self.diffxx[az_step],
                                                 self.diffyy[az_step],
                                                 self.diffxy[az_step]))
                        scene_vv[az_step] += Esn_sp
                NRCS_avg_hh[az_step] += (np.sum(np.abs(Esn_sp)**2) / surface_area)
                NRCS_avg_vv[az_step] += NRCS_avg_hh[az_step]

            # Bragg
            if scat_bragg_enable:
                if (t_now - t_last_rcs_bragg) > ocean_dt:

                    if scat_bragg_model == 'romeiser97':
                        if pol == 'DP':
                            RCS_bragg_hh, RCS_bragg_vv = self.rcs_bragg.rcs(az_rad,
                                                                            self.diffx[az_step],
                                                                            self.diffy[az_step])
                        elif pol == 'hh':
                            RCS_bragg_hh = self.rcs_bragg.rcs(az_rad,
                                                              self.diffx[az_step],
                                                              self.diffy[az_step])
                        else:
                            RCS_bragg_vv = self.rcs_bragg.rcs(az_rad,
                                                              self.diffx[az_step],
                                                              self.diffy[az_step])

                    if use_hmtf:
                        # Fix Bad MTF points
                        (self.h_mtf[az_step])[np.where(self.surface.hMTF < -1)] = -1
                        if do_hh:
                            RCS_bragg_hh[0] *= (1 + self.h_mtf[az_step])
                            RCS_bragg_hh[1] *= (1 + self.h_mtf[az_step])
                        if do_vv:
                            RCS_bragg_vv[0] *= (1 + self.h_mtf[az_step])
                            RCS_bragg_vv[1] *= (1 + self.h_mtf[az_step])

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
                cap_phase = (2 * k0) * self.t_step * c_b * (az_step + 1)
                phase_bragg[0] = surf_phase - cap_phase  # + dop_phase_p
                phase_bragg[1] = surf_phase + cap_phase  # + dop_phase_m
                bragg_scats[0] = rndscat_m.scats(t_now)
                bragg_scats[1] = rndscat_p.scats(t_now)

                if do_hh:
                    scene_hh[az_step] += ne.evaluate('sum(scat_bragg_hh * exp(1j*phase_bragg) * bragg_scats, axis=0)')
                if do_vv:
                    scene_vv[az_step] += ne.evaluate('sum(scat_bragg_vv * exp(1j*phase_bragg) * bragg_scats, axis=0)')

        v_r = (self.surface.Vx * np.sin(inc) * np.cos(az_rad) +
               self.surface.Vy * np.sin(inc) * np.sin(az_rad) -
               self.surface.Vz * np.cos(inc))

        sigma_v_r = np.std(v_r)
        # Some stats
        def weighted_stats(v, w):
            wm = np.sum(v * w) / np.sum(w)
            wsigma = np.sqrt(np.sum(w * (v - wm)**2) / np.sum(w))
            return wm, wsigma

        if do_hh:
            w = np.abs(scene_hh[0])**2
            # v_r_whh = np.sum(v_r * w) / np.sum(w)
            v_r_whh, sigma_v_r_whh = weighted_stats(v_r, w)
            # FIXME, for now just one lag

            v_ati_hh = -(np.angle(np.mean(scene_hh[1] * np.conj(scene_hh[0]))) /
                         self.t_step *const.c / 2 / self.f0 / 2 / np.pi)
            surfcoh_ati_hh = (np.mean(scene_hh[1] * np.conj(scene_hh[0]))/
                              np.sqrt(np.mean(np.abs(scene_hh[1])**2) * np.mean(np.abs(scene_hh[0])**2)))
        if do_vv:
            w = np.abs(scene_vv[0])**2
            # v_r_wvv = np.sum(v_r * w) / np.sum(w)
            v_r_wvv, sigma_v_r_wvv = weighted_stats(v_r, w)
            v_ati_vv = -(np.angle(np.mean(scene_vv[1] * np.conj(scene_vv[0]))) /
                         self.t_step * const.c / 2 / self.f0 / 2 / np.pi)
            surfcoh_ati_vv = (np.mean(scene_vv[1] * np.conj(scene_vv[0])) /
                              np.sqrt(np.mean(np.abs(scene_vv[1]) ** 2) * np.mean(np.abs(scene_vv[0]) ** 2)))

        if do_hh and do_vv:
            return {'v_r': v_r, 'scene_hh': scene_hh, 'scene_vv': scene_vv,
                    'NRCS_hh': NRCS_avg_hh, 'NRCS_vv': NRCS_avg_vv,
                    'v_r_whh': v_r_whh, 'v_r_wvv': v_r_wvv,
                    'v_ATI_hh': v_ati_hh, 'v_ATI_vv': v_ati_vv,
                    'sigma_v_r': sigma_v_r, 'sigma_v_r_whh': sigma_v_r_whh, 'sigma_v_r_wvv': sigma_v_r_wvv,
                    'surfcoh_ati_hh': surfcoh_ati_hh, 'surfcoh_ati_vv': surfcoh_ati_vv}
            # return (cfg, self.surface.Dz, v_r, scene_hh, scene_vv)
        elif do_hh:
            return {'v_r': v_r, 'scene_hh': scene_hh, 'scene_vv': None,
                    'NRCS_hh': NRCS_avg_hh, 'NRCS_vv': None,
                    'v_r_whh': v_r_whh, 'v_r_wvv': None,
                    'v_ATI_hh': v_ati_hh, 'v_ATI_vv': None,
                    'sigma_v_r': sigma_v_r, 'sigma_v_r_whh': sigma_v_r_whh,
                    'surfcoh_ati_hh': surfcoh_ati_hh}
            # return (cfg, self.surface.Dz, v_r, scene_hh)
        else:
            return {'v_r': v_r, 'scene_hh': None, 'scene_vv': scene_vv,
                    'NRCS_hh': None, 'NRCS_vv': NRCS_avg_vv,
                    'v_r_whh': None, 'v_r_wvv': v_r_wvv,
                    'v_ATI_hh': None, 'v_ATI_vv': v_ati_vv,
                    'sigma_v_r': sigma_v_r, 'sigma_v_r_wvv': sigma_v_r_wvv,
                    'surfcoh_ati_vv': surfcoh_ati_vv}
            # return (cfg, self.surface.Dz, v_r, scene_vv)

    def gmf(self, inc_deg, naz=12):
        """
        
        :param inc_deg: incident angle or array of incident angles
        :param naz: Number of azimuth angles
        :return: a namedtuple with the following fields: 'U10' 'azimuth', 'incident', 'NRCS_hh', 'NRCS_vv', 'v_r_whh',
                 'v_r_wvv', 'v_ATI_hh', 'v_ATI_vv'
        """
        incs = np.array([inc_deg]).flatten()

        azs = np.arange(naz) * 360/naz
        nrcs_hh = np.zeros((incs.size, naz))
        nrcs_vv = np.zeros_like(nrcs_hh)
        v_r_whh = np.zeros_like(nrcs_hh)
        v_r_wvv = np.zeros_like(nrcs_hh)
        v_ati_hh = np.zeros_like(nrcs_hh)
        v_ati_vv = np.zeros_like(nrcs_hh)
        sigma_v_r = np.zeros_like(nrcs_hh)
        sigma_v_r_whh = np.zeros_like(nrcs_hh)
        sigma_v_r_wvv = np.zeros_like(nrcs_hh)
        surfcoh_hh = np.zeros_like(nrcs_hh)
        surfcoh_vv = np.zeros_like(nrcs_hh)
        for inc_ind in range(incs.size):
            self.set_inc(incs[inc_ind])
            print("Incident angle: %4.2f degree" % (incs[inc_ind]))
            for az_ind in range(naz):
                res = self.surface_S(azs[az_ind])
                nrcs_hh[inc_ind, az_ind] = res["NRCS_hh"].mean()
                nrcs_vv[inc_ind, az_ind] = res["NRCS_vv"].mean()
                v_r_whh[inc_ind, az_ind] = res["v_r_whh"]
                v_r_wvv[inc_ind, az_ind] = res["v_r_wvv"]
                v_ati_hh[inc_ind, az_ind] = res["v_ATI_hh"]
                v_ati_vv[inc_ind, az_ind] = res["v_ATI_vv"]
                sigma_v_r[inc_ind, az_ind] = res["sigma_v_r"]
                sigma_v_r_whh[inc_ind, az_ind] = res["sigma_v_r_whh"]
                sigma_v_r_wvv[inc_ind, az_ind] = res["sigma_v_r_wvv"]
                surfcoh_hh[inc_ind, az_ind] = np.abs(res["surfcoh_ati_hh"])
                surfcoh_vv[inc_ind, az_ind] = np.abs(res["surfcoh_ati_vv"])
        return GMF(self.wind_u, azs, incs, nrcs_hh, nrcs_vv, v_r_whh, v_r_wvv, v_ati_hh, v_ati_vv,
                   sigma_v_r, sigma_v_r_whh, sigma_v_r_wvv, surfcoh_hh, surfcoh_vv)


def view_surface(radsurf, rel, ml=2):
    """
    
    :param rel: realization of radar surface
    :param ml: 
    :return: 
    """
    w_h = radsurf.dz[0]
    v_r = rel["v_r"][0]
    cfg = radsurf.cfg
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
    sigma = (utils.smooth(np.abs(rel["scene_hh"][0])**2, ml) /
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


def v_r_stats(rsurf, rel, ml=1):
    """
    Taking the tuple generated by surface_S, it computes the *true* radial velocity histogram as well as the
     NRCS weighted histograms. The output illustrates the (wind) wave induced bias of the Doppler.
    :param S: tuple generated by surface_S
    :param ml: spacial averaging factor (wich currently does nothing)
    """
    # FIXME
    # implement or remove ml
    v_r = rel["v_r"]
    sigma_V = np.abs(rel["scene_vv"][0])**2
    sigma_H = np.abs(rel["scene_hh"][0])**2
    vmax = np.abs(rel["v_r"]).max()
    h_v_r, bins = np.histogram(v_r, bins=np.int(2 * vmax * 40), density=True)
    h_v_r_Vw, kk = np.histogram(v_r, bins=np.int(2 * vmax * 40), density=True,
                                weights=sigma_V)
    h_v_r_Hw, kk = np.histogram(v_r, bins=np.int(2 * vmax * 40), density=True,
                                weights=sigma_H)
    bins_ = (bins[0:-1] + bins[1:]) / 2
    plt.figure()
    plt.plot(bins_, h_v_r, label='True')
    plt.plot(bins_, h_v_r_Vw, label='$\sigma_{VV}$ Weighted')
    plt.plot(bins_, h_v_r_Hw, label='$\sigma_{HH}$ Weighted')
    plt.xlabel("$v_r$ [m/s]")
    plt.ylabel("pdf($v_r$)")
    plt.legend()


if __name__ == '__main__':
    cfg_file = '/Users/plopezdekker/DATA/OCEANSAR/PAR/SKIM_proxy.cfg'
    cfg_file = '/Users/plopezdekker/DATA/OCEANSAR/PAR/S1_TOPS_emu1.cfg'
    #import oceansar.scatstats as ocs
    import drama.oceans.cmod5n as cm
    U = 8
    radsurf_U8 = RadarSurface(cfg_file, winddir=0, U10=U, t_step=1e-3)
    gmf = radsurf_U8.gmf([30], 18)

    plt.figure()


    plt.plot(gmf.azimuth, 10 * np.log10(gmf.NRCS_vv[0]), label='30')
    plt.plot(gmf.azimuth, 10 * np.log10(gmf.NRCS_vv[1]), label='12')
    #plt.plot(gmf.azimuth, 10 * np.log10(gmf.NRCS_vv[2]), label='45')
    az = np.linspace(0, 360, 100)
    plt.plot(az, 10 * np.log10(cm.cmod5n_forward(8, az + 180, np.array([6]))), 'b--')
    plt.plot(az, 10 * np.log10(cm.cmod5n_forward(8, az + 180, np.array([12]))), 'g--')
    #plt.plot(az, 10 * np.log10(cm.cmod5n_forward(8, az + 180, np.array([45]))), 'r--')
    #plt.ylim((-25, -0))
    plt.grid(True)
    plt.legend()
    plt.title("U=8 m/s")
    plt.xlabel("Azimuuth [deg]")
    plt.xlabel("Azimuth [deg]")
    plt.ylabel("$\sigma_{0,VV}$")
    plt.ylabel("$\sigma_{0,VV}$ [dB]")
    plt.figure()
    plt.plot(gmf.azimuth, gmf.v_r_wvv[0], 'b--', label='6')
    plt.plot(gmf.azimuth, gmf.v_r_wvv[1], 'g--', label='12')
    #plt.plot(gmf.azimuth, gmf.v_r_wvv[2], 'r--', label='45')
    plt.plot(gmf.azimuth, gmf.v_ATI_vv[0], 'b')
    plt.plot(gmf.azimuth, gmf.v_ATI_vv[1], 'g')
    #plt.plot(gmf.azimuth, gmf.v_ATI_vv[2], 'r')
    plt.ylabel("$v_{Dop,VV}$ [m/s]")
    plt.xlabel("Azimuth [deg]")
    plt.title("U=8 m/s")
    plt.grid(True)

    plt.figure()
    v2dop = radsurf_U8.f0/3e8 * 2
    plt.plot(gmf.azimuth, v2dop * gmf.v_r_wvv[0], 'b--', label='6')
    plt.plot(gmf.azimuth, v2dop * gmf.v_r_wvv[1], 'g--', label='12')
    # plt.plot(gmf.azimuth, gmf.v_r_wvv[2], 'r--', label='45')
    plt.plot(gmf.azimuth, v2dop * gmf.v_ATI_vv[0], 'b')
    plt.plot(gmf.azimuth, v2dop * gmf.v_ATI_vv[1], 'g')
    # plt.plot(gmf.azimuth, gmf.v_ATI_vv[2], 'r')
    plt.ylabel("$f_{Dop,VV}$ [Hz]")
    plt.xlabel("Azimuth [deg]")
    plt.title("U=8 m/s")
    plt.grid(True)