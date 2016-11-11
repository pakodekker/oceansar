import numpy as np
from scipy import interpolate
from collections import namedtuple
#from drama.geo import orbit_to_vel
from oceansar import constants as const
#import drama.utils.gohlke_transf as trans
#from drama.utils.coord_trans import (rot_z, rot_z_prime)

def orbit_to_vel(orbit_alt, ground=False,
                 r_planet=const.r_earth,
                 m_planet=const.m_earth):
    """ Calculates orbital/ground velocity assuming circular orbit

        :param orbit_alt: Satellite orbit altitude
        :param ground: If true, returned value will be ground velocity

        :returns: Orbital or Ground velocity
    """
    v = np.sqrt(const.G * m_planet/(r_planet + orbit_alt))

    # Convert to ground velocity if needed
    if ground:
        v = r_planet / (r_planet + orbit_alt) * v

    return v


def inc_to_sr(theta_i, orbit_alt, r_planet=const.r_earth):
    """ Calculates slant range angle given incidence angle

        :param theta_i: Incidence angle
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: Slant range
    """

    theta_l = inc_to_look(theta_i, orbit_alt, r_planet=r_planet)
    delta_theta = theta_i - theta_l

    return np.sqrt((orbit_alt + r_planet -
                    r_planet * np.cos(delta_theta))**2 +
                   (r_planet * np.sin(delta_theta))**2)


def inc_to_gr(theta_i, orbit_alt, r_planet=const.r_earth):
    """ Calculates incidence angle given ground range

        :param theta: Incidence angle
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: Ground range
    """
    return r_planet * (theta_i - inc_to_look(theta_i,
                                             orbit_alt,
                                             r_planet=r_planet))


def inc_to_look(theta_i, orbit_alt, r_planet=const.r_earth):
    """ Calculates look angle given incidence angle

        :param theta_i: Incidence angle [rad]
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: Look angle [rad]
    """

    return np.arcsin(np.sin(theta_i)/(r_planet + orbit_alt) * r_planet)


def look_to_inc(theta_l, orbit_alt, r_planet=const.r_earth):
    """ Calculates incidence angle given look angle

        :param theta_l: Look angle
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: Incidence angle
    """
    return np.arcsin(np.sin(theta_l)*(r_planet + orbit_alt)/r_planet)


def look_to_sr(theta_l, orbit_alt, r_planet=const.r_earth):
    """ Calculates slant range angle given look angle

        :param theta_l: Look angle
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: slant range
    """
    theta_i = look_to_inc(theta_l, orbit_alt, r_planet=r_planet)
    delta_theta = theta_i - theta_l
    return np.sqrt((orbit_alt + r_planet -
                    r_planet * np.cos(delta_theta))**2 +
                   (r_planet * np.sin(delta_theta))**2)


def sr_to_geo(slant_range, orbit_alt,
              r_planet=const.r_earth,
              m_planet=const.m_earth):
    """ Calculates zero Dopplerinterpolated SAR geometric parameters given a
        set of slant range points

        :param slant_range: Set of ground range points
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius
        :param m_planet: mass of planet, defaults to Earth's mass

        :returns: ground range, incidence angle, look angle
    """
    # Calculate look/incident angles
    theta_l = np.linspace(0, max_look_angle(orbit_alt, r_planet=r_planet),
                          500)
    theta_i = look_to_inc(theta_l, orbit_alt, r_planet=r_planet)
    delta_theta = theta_i - theta_l
    r_track = np.cos(delta_theta) * r_planet
    v_orb = orbit_to_vel(orbit_alt, r_planet=r_planet, m_planet=m_planet)
    b = v_orb**2 * r_track / (r_planet + orbit_alt)

    # Calculate Ground Range and Slant Range
    gr = r_planet * delta_theta
    sr = np.sqrt((orbit_alt + r_planet - r_planet * np.cos(delta_theta))**2 +
                 (r_planet * np.sin(delta_theta))**2)

    # Interpolate look/incidence angles and Slant Range
    theta_l_interp = interpolate.InterpolatedUnivariateSpline(sr, theta_l, k=2)(slant_range)
    theta_i_interp = interpolate.InterpolatedUnivariateSpline(sr, theta_i, k=2)(slant_range)
    gr_interp = interpolate.InterpolatedUnivariateSpline(sr, gr, k=2)(slant_range)
    b_interp = interpolate.InterpolatedUnivariateSpline(sr, b, k=2)(slant_range)
    return (gr_interp, theta_i_interp, theta_l_interp, b_interp)


def gr_to_geo(ground_range, orbit_alt, r_planet=const.r_earth):
    """ Calculates interpolated SAR geometric parameters given a set of
        ground range points

        :param ground_range: Set of ground range points
        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: slant range, incidence angle, look angle
    """
    # Calculate look/incident angles
    theta_l = np.linspace(0, max_look_angle(orbit_alt), 500)
    theta_i = look_to_inc(theta_l, orbit_alt, r_planet=r_planet)
    delta_theta = theta_i - theta_l

    # Calculate Ground Range and Slant Range
    gr = r_planet * delta_theta
    sr = np.sqrt((orbit_alt + const.r_earth -
                  r_planet * np.cos(delta_theta))**2 +
                 (r_planet * np.sin(delta_theta))**2)

    # Interpolate look/incidence angles and Slant Range
    theta_l_interp = interpolate.InterpolatedUnivariateSpline(gr, theta_l, k=2)(ground_range)
    theta_i_interp = interpolate.InterpolatedUnivariateSpline(gr, theta_i, k=2)(ground_range)
    sr_interp = interpolate.InterpolatedUnivariateSpline(gr, sr, k=2)(ground_range)

    return sr_interp, theta_i_interp, theta_l_interp


def max_look_angle(orbit_alt, r_planet=const.r_earth):
    """ Calculates maximum look angle given satellite orbit altitude

        :param orbit_alt: Satellite orbit altitude
        :param r_planet: radious of planet, defaults to Earth's radius

        :returns: Maximum look angle
    """
    return np.arcsin(r_planet / (r_planet + orbit_alt))



