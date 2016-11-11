"""
    **Custom Added Constants**

     :r_earth:        Earth radius [m]
     :m_earth:        Earth mass [Kg]
     :c:              speed of light [m/s]
     :g_star:         gravitational constant g_star [m^3/km/s^2]
     :gm_earth:       g_star * mass of earth [m^3/s^2]
     :gm_sun:         g_star * mass of sun [m^3/s^2]
     :au:             astronomical unit [m]
     :r_sun:          radius of the sun [m]
     :p_sun:          radiation pressure of the sun [N/m^2]
     :gm_moon:        g_star * mass of moon [m^3/s^2]
     :d_moon_earth:   mean distance moon to earth [m]
     :r_moon:         radius of moon [m]
     :h:              Planck's constant [Js]
     :omega_earth:    angular velocity of earth [1/s] WGS84
     :pi:             pi to 30 digits
     :j2:             zonal coefficient
     :j3:             zonal coefficient
     :j4:             zonal coefficient
     :j5:             zonal coefficient

 """

import numpy as np
from scipy.constants import *

r_earth = 6378.137e3  # Earth equat. radius [m]
m_earth = 5.9742e24  # Earth mass [Kg]
c = 2.99792458e8  # speed of light [m/s]
g_star = 6.673e-11  # gravitational constant g_star [m^3/km/s^2]
gm_earth = 398600.4415e9  # g_star * mass of earth [m^3/s^2]
gm_sun = 1.32712440018e20  # g_star * mass of sun [m^3/s^2]
au = 149597870.691e3  # astronomical unit [m]
r_sun = 6.96e8  # radius of the sun [m]
p_sun = 4.560e-6  # radiation pressure of the sun [N/m^2]
gm_moon = 4902.801e9  # g_star * mass of moon [m^3/s^2]
d_moon_earth = 384400e3  # mean distance moon to earth [m]
r_moon = 1738e3  # radius of moon [m]
h = 6.626068765e-34  # Planck's constant [Js]
omega_earth = 0.729211585530e-4  # angular velocity of earth [1/s] WGS84
pi = 3.141592653589793238462643383279  # pi to 30 digits
j2 = 0.00108263  # zonal coefficient
j3 = -0.00254e-3  # zonal coefficient
j4 = -0.00161e-3  # zonal coefficient
j5 = -0.246e-6  # zonal coefficient

# Sea water rel.dielectric constant
epsilon_sw = np.complex(73, 18)
# Water refractive index (approximate)
n_w = 4./3.
# Impedance of vacuum [Ohm]
etha_0 = physical_constants['characteristic impedance of vacuum'][0]


def r_equatorial(system='wgs84'):
    """ Defines equatorial radius according to a specific system

        :date: 20.10.2014

        :author: Jalal Matar

        :param system: defines the measurments
            system (iau, krassowsky or wgs84)

        :returns: equatorial radius
    """

    if (system == 'iau'):
        r_eq = 6378.160e3

    elif (system == 'krassowski'):   # as determined by Krassowsky
        r_eq = 6378.245e3

    elif (system == 'wgs84'):
        r_eq = 6378.137e3  # WGS 84, IAG-GRS 80,used by ICAO since 1998

    return r_eq


def r_polar(system='wgs84'):
    """ Defines polar radius according to a specific system

        :date: 20.10.2014

        :author: Jalal Matar

        :param system: defines the measurments
          system (iau, krassowsky or wgs84)

        :returns: polar radius
    """

    if (system == 'iau'):
        r_po = 6356.775e3  # m, f = 1/297.0

    elif (system == 'krassowski'):  # as determined by Krassowsky
        r_po = 6356.86301877e3  # m, f = 3.35232986926e-3

    elif (system == 'wgs84'):
        r_po = 6356.75231425e3  # m, f = 1/298.257222101

    return r_po
