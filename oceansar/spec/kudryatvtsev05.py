import numpy as np
from oceansar import constants as const
from matplotlib import pyplot as plt

__author__ = "Paco Lopez Dekker"
__email__ = "F.LopezDekker@tudeft.nl"


def beta(k, U, phi=0, z_0=0.0002):
    """
    Equation (17) in Kudryatvtsev 2005
    
    :param k: 
    :param U: 
    :param phi: 
    :param z_0: Roughness scale, set to 0.0002 following Wikipedia
    :return: 
    """

    # Phase velocity
    c = np.sqrt(const.g/k + 0.072e-3 * k)
    #
    rho_water = 1e3 # kg /mˆ3
    rho_air = 1.225 # kg /mˆ3
    kappa = 0.4
    #
    c_b = 1.5 * (rho_air/rho_water) * (np.log(np.pi/(k * z_0))/kappa -c/U)
    # (17)
    return c_b * (U / c)**2 * np.cos(phi) * np.abs(np.cos(phi))


def beta_v(k, U, phi=0, z_0=0.0002, v=1.15e-6):
    """
    # After (16) 
    :param k: 
    :param U: 
    :param phi: 
    :param z_0: 
    :param v: viscosity coefficient of sea water [m^2/s]
    :return: 
    """
    gamma = 0.07275/1e3
    omega = np.sqrt(const.g * k + gamma * k**3)
    b_v = beta(k, U, phi, z_0) - 4 * v * k**2 / omega
    return np.where(b_v > 0, b_v, 0)


def B0(k, U, phi=0, z_0=0.0002, v=0.0013, alpha=5e-3, n=5):
    """
    Equation (24)
    :param k: 
    :param U: 
    :param phi: 
    :param z_0: 
    :param v: viscosisty coefficient of water
    :param alpha: 
    :return: 
    """
    return alpha * beta_v(k, U, phi, z_0, v)**(1/n)


def I_sw(k:)
