import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import linspace
from scipy import pi, sqrt, exp
from scipy.special import erf, gamma
from scipy import interpolate as interp
import pickle
import datetime
import os
import numpy.ma as ma
import bisect
from pandas import DataFrame

"""
This file contains code to read buoy wave spectra (previously pickled) and transform it into a a directional
wave-spectrum. 
@author:Martijn Kwant and Paco López Dekker
"""


def alpha(omega, depth):
    """pp 124 Eq 5.4.21"""
    g = 9.81
    a = (omega ** 2 * depth / g)
    return a


def beta(a):
    b = a * (np.tanh(a)) ** (-1 / 2)
    return b


def n(k_d):
    """calculate n"""
    n_i = (1 + ((2 * k_d) / np.sinh(2 * k_d))) / 2
    return n_i


def kd(omega, depth):
    """pp 124 Eq 5.4.22"""
    a = alpha(omega, depth)
    b = beta(a)

    for x in range(1):
        k_d = (a + b ** 2 * (np.cosh(b)) ** -2) / \
              (np.tanh(b) + b * (np.cosh(b)) ** -2)
        a = k_d
        b = beta(a)
    return k_d


def c(omega, k_d):
    """pp. 125 Eq 5.4.23"""
    g = 9.81
    c_i = (g / omega) * np.tanh(k_d)
    return c_i


def cg(w1, w2, k1, k2):
    """pp. 127, Eq 5.4.30"""
    c_g = (np.abs(w1 - w2) / 2) / (np.abs(k1 - k2) / 2)
    return c_g


class BuoySpectra():
    def __init__(self, bdata, depth=26.7, heading=-11.2):
        """
        
        :param bdata: 
        :param depth: 
        :param heading: heading of SAR, East of North (in degree)
        """
        self.bdata = bdata
        self.heading = heading
        self.depth = depth
        # Just to know the are there
        self.Sk = lambda k: 0
        self.kmin = 0.01
        self.kmax = 0.1
        self.k = np.array([0.01, 0.1])
        self.init_buoy2Sk(self.depth)
        # We rotate the direction so that it goes in the direction of the wave propagation
        self.dir = interp.interp1d(self.k, np.angle(np.exp(1j * (np.radians(bdata[2]) - np.pi))),
                                   bounds_error=False, fill_value=0,
                                   kind='nearest')
        self.spread = interp.interp1d(self.k, np.radians(bdata[3]), bounds_error=False, fill_value=1)

    def init_buoy2Sk(self, depth=26.7):
        bdata = self.bdata
        freq = bdata[0]
        fstart = freq - 0.005
        fstart[0] = freq[0]
        fend = freq + 0.005
        fend[-1] = freq[-1]
        fbin = fend - fstart
        f_vec = np.vstack((fstart, fend, fbin))
        Ef_1d = bdata[1]
        Eftot = np.sum(Ef_1d * f_vec[2])
        print('Total Ef spec = ' + str(Eftot) + ' [m2]')

        # calculate rad f spectrum
        Ew_spec = Ef_1d / (2 * np.pi)
        w_vec = f_vec * 2 * np.pi
        Ew_tot = np.sum(Ew_spec * w_vec[2])
        print('Total Ew spec = ' + str(Ew_tot) + ' [m2]')
        k_1 = kd(w_vec[0], depth) / depth
        k_2 = kd(w_vec[1], depth) / depth
        k_vec = np.vstack((k_1, k_2, k_2 - k_1))
        dwdk = w_vec[2] / k_vec[2]
        Ek_spec = Ew_spec * dwdk
        self.k = (k_1 + k_2) / 2
        self.Sk = interp.interp1d(self.k, Ek_spec, bounds_error=False, fill_value=0)
        self.kmin = self.k.min()
        self.kmax = self.k.max()

    def dirspread(self, k, theta):
        wtheta = np.angle(np.exp(1j * (theta - self.dir(k))))
        s = 2 / self.spread(k)**2 - 1
        # (2 / spr_i) - 1
        D = 2 ** (2 * s - 1) / np.pi * gamma(s + 1.) ** 2 / gamma(2. * s + 1.) * np.cos(wtheta / 2.) ** (2. * s)
        return D

    def Sk2(self, kx, ky):
        th = np.arctan2(ky, kx) + np.radians(self.heading)
        k = np.sqrt(kx**2 + ky**2)
        k_inv = np.where(k != 0, 1/k, 0)
        return self.Sk(k) * k_inv * self.dirspread(k, th)


def load_buoydata(file, date=None):
    """    
    :param file: npz file with buoy data
    :param date: Optional datetime.datetime variable, if given it looks for the data closest to that date
    :return: 
    """
    data = np.load(file, encoding='bytes')
    dates = data['dates']
    numd = dates.size
    arr0 = data['arr0']
    shp = (numd,) + arr0.shape
    data_all = np.zeros(shp)
    data_all[0, :, :] = arr0
    for ind in range(1, numd):
        data_all[ind, :, :] = data[('arr%i' % ind)]
    if date is None or type(date) is not datetime.datetime:
        return dates, data_all
    else:
        ind = np.argmin(np.abs(dates - date))
        dmin = (dates[ind] - date).total_seconds()/60
        if np.abs(dmin) > 10:
            print('load_buoydata: offset with respect to target time is', (dates[ind] - date))
        return dates[ind], data_all[ind]

    # ~~~~~~ Execute ~~~~~~
if __name__ == '__main__':

    locpath = "/Users/plopezdekker/DATA/OCEANSAR/BuouyData/M170513184/out/buoyspectra_k13.npz"
    # load data
    tdate = datetime.datetime(2015,4,26,17,25,21)
    dates, data = load_buoydata(locpath)
    date, data = load_buoydata(locpath, dates[5])
    bS = BuoySpectra(data, heading=-11.2)
    kx = np.linspace(-bS.kmax, bS.kmax, 1001).reshape((1, 1001))
    ky = np.linspace(-bS.kmax, bS.kmax, 1001).reshape((1001, 1))
    S2 = bS.Sk2(kx, ky)
    plt.figure()
    plt.imshow(S2, origin='lower', extent=[-bS.kmax, bS.kmax, -bS.kmax, bS.kmax])
    plt.xlim((-0.2, 0.2))
    plt.ylim((-0.2, 0.2))
    # kx, ky, E_kxky = transform_spectra(data, depth=26.7, rot_deg=-11.)
    # plot_Ek_spec(deg, k_new, Edir_k)
    # plot_kxky_spec(k_loc, kx, ky, E_kxky)

