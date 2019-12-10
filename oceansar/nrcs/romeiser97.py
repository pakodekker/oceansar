
import numpy as np
from scipy import interpolate

from oceansar import spec
from oceansar import spread

from oceansar import constants as const
import numexpr as ne

class RCSRomeiser97():
    """ Bragg scattering model (R. Romeiser '97)

        This function returns RCS using the model described in the paper titled:
        "An improved composite surface model for the radar backscattering cross
        section of the ocean surface", R. Romeiser and W. Alpers (1997)

        :param k0: Radar wave number
        :param inc_angle: Nominal incidence angle
        :param pol: Polarization ('vv', 'hh')
        :param dx: Surface facets X size
        :param dy: Surface facets Y size
        :param wind_U: Wind speed (U_10)
        :param wind_dir: Wind blowing direction (rad)
        :param wind_fetch: Wind fetch
        :param sw_spec_func: Omnidirectional spectrum function
        :param sw_spread_func: Spreading function
        :param d: Filter threshold (Kd = Kb/d)
    """

    def __init__(self, k0, inc_angle, pol, dx, dy,
                 wind_U, wind_dir, wind_fetch,
                 sw_spec_func, sw_spread_func, d):

        # Save parameters
        self.wind_U = wind_U
        self.wind_dir = wind_dir
        self.wind_fetch = wind_fetch
        self.dx = dx
        self.dy = dy
        self.pol = pol
        self.k0 = k0
        self.inc_angle = inc_angle
        self.d = d
        self.sw_spec_func = sw_spec_func
        self.sw_dir_func = sw_spread_func

        # Initialize Romeiser Interpolator (to increase speed!)
        # TODO: Initialize interpolator ONLY in Kb region
        #use always interpolator
        #if self.sw_spec_func == 'romeiser97':
        k = np.arange(1, 2*k0, 0.1)
        spectrum = self.interpolator = spec.models[self.sw_spec_func](k, self.wind_U, self.wind_fetch)
        #self.spec_interpolator = interpolate.InterpolatedUnivariateSpline(k, spectrum, k=1)
        self.spec_interpolator = interpolate.interp1d(k, spectrum,
                                                      bounds_error=False,
                                                      fill_value=1  )
        if (self.sw_dir_func == 'none') and (self.sw_spec_func == 'elfouhaily'):
            raise Exception('Elfouhaily spectrum requires a direction function')

    def rcs(self, az_angle, diffx, diffy):
        """ Returns RCS map of a surface given its geometry

            :param az_angle: Azimuth angle
            :param diffx: Surface X slope
            :param diffy: Surface Y slope
        """

        # Parallel (s_p) and orthogonal (s_t) slopes
        cos_az = np.cos(az_angle)
        sin_az = np.sin(az_angle)
        # s_p = np.arctan(cos_az * diffx + sin_az * diffy)
        # s_t = np.arctan(-sin_az * diffx + cos_az * diffy)
        s_p = ne.evaluate("arctan(cos_az * diffx + sin_az * diffy)")
        s_t = ne.evaluate("arctan(-sin_az * diffx + cos_az * diffy)")

        # Effective local incidence angle (6)
        cos_s_t = np.cos(s_t)
        cos_s_t_2 = cos_s_t**2
        sin_s_t_2 = 1 - cos_s_t_2
        sin_inc_anglep = np.sin(self.inc_angle - s_p)
        cos_inc_anglep = np.cos(self.inc_angle - s_p)
        cos_inc_anglep_2 = cos_inc_anglep**2
        sin_inc_anglep_2 = sin_inc_anglep**2
        cos_theta_i = cos_inc_anglep * cos_s_t
        sin_theta_i_2 = 1 - cos_theta_i**2
        #theta_i = np.arccos(np.cos(self.inc_angle - s_p) * np.cos(s_t))

        # Complex scattering coefficient (3a, 3b)
        b_hh = const.epsilon_sw / (cos_theta_i + np.sqrt(const.epsilon_sw))**2.
        b_vv = (((const.epsilon_sw**2.) * (1. + sin_theta_i_2)) /
                ((const.epsilon_sw * cos_theta_i + np.sqrt(const.epsilon_sw))**2.))

        # Weighting function (10) & Dimensions to mace RCS (not NRCS)
        w = cos_inc_anglep/(np.cos(self.inc_angle) * np.cos(s_p))
        # T(s_p, s_n) (5)
        sin_s_t_2_over_sin_theta_i_2 = sin_s_t_2 / sin_theta_i_2
        sin_inc_anglep_2_x_cos_s_t_2_over_sin_theta_i_2 = sin_inc_anglep_2 * cos_s_t_2 / sin_theta_i_2
        F0 = 8 * np.pi * (self.k0 * cos_theta_i)**4
        if self.pol == 'vv' or self.pol == 'DP':
            T_vv = (F0 *
                    np.abs((sin_inc_anglep_2_x_cos_s_t_2_over_sin_theta_i_2 * b_vv) + (sin_s_t_2_over_sin_theta_i_2 * b_hh))**2.)
            T_vv = w * T_vv
        if self.pol == 'hh' or self.pol == 'DP':
            T_hh = (F0 *
                    np.abs((sin_inc_anglep_2_x_cos_s_t_2_over_sin_theta_i_2 * b_hh) + (sin_s_t_2_over_sin_theta_i_2 * b_vv))**2.)
            T_hh = w * T_hh

        # T = T*w

        # Bragg wave number magnitude and direction (7) (8)
        k_b = 2.*self.k0*np.sqrt(sin_inc_anglep_2 + (cos_inc_anglep_2*sin_s_t_2))
        k_b = np.where(k_b > (2 * self.k0), 0, k_b)
        phi_b = az_angle + np.arctan((cos_inc_anglep * np.sin(-s_t)) / np.sin(self.inc_angle - s_p))

        # Calculate folded spectrum
        k_inp = np.array([k_b, k_b])

        theta_inp = np.array([phi_b, phi_b + np.pi]) - self.wind_dir
        theta_inp = np.angle(np.exp(1j * theta_inp))
        #if self.sw_spec_func == 'romeiser97':
        spectrum_1D = self.spec_interpolator(k_b.flatten()).reshape((1,) +
                                                                    k_b.shape)
        #else:
        #    spectrum = spec.models[self.sw_spec_func](k_inp, self.U_10, self.fetch)

        spectrum = (spectrum_1D / k_inp *
                    spread.models[self.sw_dir_func](k_inp, theta_inp,
                                                    self.wind_U,
                                                    self.wind_fetch))

        # Calculate RCS & Filter result
        # tan_inc_angle_md = np.tan(self.inc_angle - self.d/2.)
        # tan_inc_angle_pd = np.tan(self.inc_angle + self.d/2.)
        # bad_ones = np.where(np.logical_or(s_p < tan_inc_angle_md, s_p > tan_inc_angle_pd))
        bad_ones = np.where(4 * sin_inc_anglep_2 < (self.d**2))
        if self.pol == 'vv':
            rcs = T_vv * spectrum * self.dx * self.dy
            # rcs[0] = np.where(((s_p > tan_inc_angle_md) &
            #                    (s_p < tan_inc_angle_pd)), 0., rcs[0])
            # rcs[1] = np.where(((s_p > tan_inc_angle_md) &
            #                    (s_p < tan_inc_angle_pd)), 0., rcs[1])
            rcs[0][bad_ones] = 0
            rcs[1][bad_ones] = 0
            return rcs
        elif self.pol == 'hh':
            rcs = T_hh * spectrum * self.dx * self.dy
            # rcs[0] = np.where(((s_p > tan_inc_angle_md) &
            #                    (s_p < tan_inc_angle_pd)), 0., rcs[0])
            # rcs[1] = np.where(((s_p > tan_inc_angle_md) &
            #                    (s_p < tan_inc_angle_pd)), 0., rcs[1])
            rcs[0][bad_ones] = 0
            rcs[1][bad_ones] = 0
            return rcs
        else:
            rcs_vv = T_vv * spectrum * self.dx * self.dy
            # rcs_vv[0] = np.where(((s_p > tan_inc_angle_md) &
            #                       (s_p < tan_inc_angle_pd)), 0., rcs_vv[0])
            # rcs_vv[1] = np.where(((s_p > tan_inc_angle_md) &
            #                       (s_p < tan_inc_angle_pd)), 0., rcs_vv[1])
            rcs_vv[0][bad_ones] = 0
            rcs_vv[1][bad_ones] = 0
            rcs_hh = T_hh * spectrum * self.dx * self.dy
            # rcs_hh[0] = np.where(((s_p > tan_inc_angle_md) &
            #                       (s_p < tan_inc_angle_pd)), 0., rcs_hh[0])
            # rcs_hh[1] = np.where(((s_p > tan_inc_angle_md) &
            #                       (s_p < tan_inc_angle_pd)), 0., rcs_hh[1])
            # rcs_hh[0][bad_ones] = 0
            rcs_hh[1][bad_ones] = 0
            return (rcs_hh, rcs_vv)

