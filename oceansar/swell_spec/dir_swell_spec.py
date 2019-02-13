# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 19:13:58 2019

@author: lyh
"""
import numpy as np
from oceansar import constants as const

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:09:40 2019
Create the directional swell spectrum by both generating the spread
and the directional distribution based on Gaussian swell spectrum

            :param wave_dir: the dirction of the swell
            :param sigf: spread in frequency
            :param freq_r: related to the wavelength of swell (lmd = g / (2pi*f^2))
            :param sigs: spread in direction
            :param Hs: significant wave height (Hs = 4std)
         """


def ardhuin_swell_spec(k_axis, theta_axis, dir_swell_dir,
                       freq_r=0.068, sigf=0.007, sigs=8, Hs=1.45):
    # the swell spectrum is given in frequency domain
    f_k = 1 / 2 / np.pi * np.sqrt(const.g * k_axis)
    # frequency spectrum (gaussian)
    fac_f_k = 1 / 4 / np.pi * np.sqrt(const.g / k_axis)
    amp = (Hs / 4) ** 2 / sigf / np.sqrt(2 * np.pi)
    efs = (amp * np.exp(-(f_k - freq_r) ** 2 / (2 * sigf ** 2)) + 1E-5) * fac_f_k
    # directional distribution
    ang = np.angle(np.exp(1j * (theta_axis - np.radians(dir_swell_dir))))
    dirss = np.exp(-ang ** 2 / (2 * np.radians(sigs) ** 2))
    # normalization: the integration of all the theta for each k is 1
    factor = np.sqrt(2 * np.pi) * np.radians(sigs)
    # final 2D spectrum
    swell_spec = efs * dirss / factor
    return swell_spec
