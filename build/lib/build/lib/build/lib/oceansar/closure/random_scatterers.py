
import numpy as np
import scipy as sp
from oceansar import utils

class randomscat_ts():
    """ A class defining a random time series of complex values with a
        Gaussian Spectrum
    """

    def __init__(self, tau, dim, Fs, N=None, seed=None):
        if not (seed is None):
            np.random.seed(seed)
        if N is None:
            N = int(tau * Fs * 200)
        self.N = utils.optimize_fftsize(N, max_prime=5)

        f = np.fft.fftfreq(self.N, 1./Fs)
        a = 1 / (tau**2)
        spec = np.sqrt(np.pi / a) * np.exp(-np.pi**2 * f**2 / a)
        s_f = (np.random.normal(size=self.N) +
               1j * np.random.normal(size=self.N))
        s = np.fft.ifft(s_f * np.sqrt(spec))
        self.s = s / np.sqrt(np.mean(np.abs(s)**2))
        self.ind0 = (N * np.random.uniform(size=dim)).astype(int)
        self.Fs= Fs

    def scats(self, t):
        tind = int(t * self.Fs)
        inds = np.mod(self.ind0 + tind, self.N)
        return self.s[inds]
