# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:12:51 2014

@author: Paco Lopez Dekker
"""


import numpy as np
# from scipy import weave
from numba import jit
import matplotlib.pyplot as plt


def calc_sinc_vec(n_sinc_samples=6, sinc_ovs=10, Fs=1.0):
    n_sinc_bins = int(sinc_ovs * n_sinc_samples + 1)
    sinc_bins = np.arange(n_sinc_bins) * 1. / sinc_ovs - n_sinc_samples/2
    sinc_vec = np.array(np.sinc(sinc_bins/Fs), dtype=np.float32)
    return sinc_vec


@jit("void(c16[:],i4[:],i4[:],f4[:],i4,i4,c16[:])", nopython=True)
def profile_integrator(scat, bin_i, bin_f, pulse, n_samp, over_samp, output):
    """Integrates contributions of all points to range profile

       :param scat: vector with complex scattering coefficients
       :param bin_i: vector of integer (floor) range indices
       :param bin_f: relative position in oversampled waveform
       :param pulse: range pulse, typically a sinc
       :param n_samp: length of not oversampled pulse
       :param over_samp: oversampling factor of pulse
       :param output: vector where generated output will be stored
    """
    npts = scat.size
    nbins = output.size
    rnbins = output.size - n_samp - 1
    for i in range(npts):
        bin_now = bin_i[i] - int(n_samp/2)
        sbin = bin_f[i]
        if (bin_now < 0) or (bin_now > rnbins):
            #We are at an edge and proceed with care
            for j in range(n_samp):
                if (bin_now >= 0) and (bin_now < nbins):
                    output[bin_now] += pulse[sbin+j*over_samp] * scat[i]
                bin_now += 1
        else:
            for j in range(n_samp):
                output[bin_now] += pulse[sbin+j*over_samp] * scat[i]
                bin_now += 1


@jit("void(c16[:, :],i4[:, :],i4[:, :],f4[:],i4,i4,c16[:, :])", nopython=True)
def profile_integrator_1d(scat, bin_i, bin_f, pulse, n_samp, over_samp, output):
    """Integrates contributions of all points to range profile

       :param scat: vector with complex scattering coefficients
       :param bin_i: vector of integer (floor) range indices
       :param bin_f: relative position in oversampled waveform
       :param pulse: range pulse, typically a sinc
       :param n_samp: length of not oversampled pulse
       :param over_samp: oversampling factor of pulse
       :param output: vector where generated output will be stored
    """
    (naz, npts) = scat.shape
    nbins = output.shape[1]
    rnbins = nbins - n_samp - 1
    for ia in range(naz):
        for i in range(npts):
            bin_now = bin_i[ia, i] - int(n_samp/2)
            sbin = bin_f[ia, i]
            if (bin_now < 0) or (bin_now > rnbins):
                #We are at an edge and proceed with care
                for j in range(n_samp):
                    if (bin_now >= 0) and (bin_now < nbins):
                        output[ia, bin_now] += pulse[sbin+j*over_samp] * scat[ia, i]
                    bin_now += 1
            else:
                for j in range(n_samp):
                    output[ia, bin_now] += pulse[sbin+j*over_samp] * scat[ia, i]
                    bin_now += 1


def chan_profile_numba(srg, scene, binsize, min_sr,
                         sinc_vec, n_sinc_samples, sinc_ovs,
                         output, rg_only=False):
    """ Calculates channel profile by summing all grid points with the same
        slant range

        :param srg: Matrix with slant range values for each grid
        :param scene: Matrix with scene
        :param output: Matrix where generated output will be stored
        :param binsize: Slant range bin resolution
        :param min: Minimum slant range value to consider
        :param max: Maximum slant range value to consider
        :param sinc_vec: oversampled sinc, the idea is to reuse it
        :param rg_only: set to do only integation in last dimension

        .. note::
            This function is implemented in C in order to achieve good
            performance
    """

    binsize = float(binsize)
    #threshold = int(threshold)
    #Fractional bin
    bin_f = (srg - min_sr) / binsize
    bin_d = np.int32(np.floor(bin_f))
    bin_f = np.int32(np.round(sinc_ovs * (1 - (bin_f - bin_d))))
    #n_lobes = 2*n_sinc_lobes
    outtmp = np.zeros(output.size, dtype=np.complex64)
    if rg_only:
        profile_integrator_1d(scene, bin_d, bin_f, sinc_vec, n_sinc_samples,
                              sinc_ovs, output)
    else:
        profile_integrator(scene, bin_d, bin_f, sinc_vec, n_sinc_samples,
                           sinc_ovs, output)


def chan_profile_weave(srg, scene, binsize, min_sr,
                      sinc_vec, n_sinc_samples, sinc_ovs,
                      output):
    """ Calculates channel profile by summing all grid points with the same
        slant range

        :param srg: Matrix with slant range values for each grid
        :param scene: Matrix with scene
        :param output: Matrix where generated output will be stored
        :param binsize: Slant range bin resolution
        :param min: Minimum slant range value to consider
        :param max: Maximum slant range value to consider
        :param sinc_vec: oversampled sinc, the idea is to reuse it

        .. note::
            This function is implemented in C in order to achieve good
            performance
    """

    # Type 'conversion' needed for weave
    binsize = float(binsize)
    #threshold = int(threshold)
    npts = srg.size
    nbins = output.size
    n_sinc_bins = sinc_ovs * n_sinc_samples + 1
    if np.size(sinc_vec) != n_sinc_bins:
        sinc_bins = np.arange(n_sinc_bins)*1./sinc_ovs-n_sinc_lobes
        sinc_vec = np.sinc(sinc_bins)
    #Fractional bin
    bin_f = (srg - min_sr) / binsize
    bin_d = (np.floor(bin_f))
    bin_f = np.floor(sinc_ovs * (1 - (bin_f - bin_d)))
    code = """
    int bin,sbin, i,j;
    int tn_sinc_lobes, rnbins;
    tn_sinc_lobes = 2 * n_sinc_lobes;
    rnbins = nbins - tn_sinc_lobes -1;

    for(i=0; i<npts; i++)
    {
        bin = (int)(bin_d[i]);
        bin = bin - n_sinc_lobes;
        sbin = (int)(bin_f[i]);
        //Check if there are edge issues
        if((bin < 0) || (bin > rnbins))
        {
            //We are at an edge and proceed with care
            for(j=0; j<2*n_sinc_lobes;j++)
            {
                if((bin > 0) && (bin < (nbins - 1)))
                {
                    output[bin] += sinc_vec[sbin+j*sinc_ovs]*scene[i];
                }
                bin++;
            }
        }
        else
        {
            //We are safe and do not need to check for boundary problems
            for(j=0; j<2*n_sinc_lobes;j++)
            {
                output[bin] += sinc_vec[sbin+j*sinc_ovs]*scene[i];
                bin++;
            }
        }
    }


    """

    weave.inline(code, ['scene', 'npts', 'nbins', 'output',
                        'bin_d', 'bin_f', 'sinc_vec', 'sinc_ovs',
                        'n_sinc_lobes'], verbose=2)


def range_profile_test(N):
    scat = np.zeros(N,dtype=np.complex)
    srg = np.random.rand(N)*100
    out1 = np.zeros(110,dtype=np.complex)
    out2 = np.zeros(110,dtype=np.complex)
    min_sr = 0.
    binsize = 1.0
    n_sinc_lobes = 3
    sinc_ovs = 10
    sincv = calc_sinc_vec(n_sinc_lobes, sinc_ovs)
    scat[100] = 1
    srg[100] = 50.5
    chan_profile_numba(srg, scat, binsize, min_sr,
                       sincv, n_sinc_lobes, sinc_ovs,
                       out1)
    plt.plot(np.linspace(0,110,110),np.real(out1))
    scat = np.random.randn(N) + 1j*np.random.randn(N)


if __name__ == '__main__':
    N=1024
    scat = np.zeros((N, N), dtype=np.complex)
    scat = np.random.randn(N, N) + 1j*np.random.randn(N, N)
    srg = np.random.rand(N, N) * 100
    out1 = np.zeros((N, 110), dtype=np.complex)
    out1f = np.zeros(110, dtype=np.complex)
    out2 = np.zeros((N, 110), dtype=np.complex)
    min_sr = 0.
    binsize = 1.0
    n_sinc_lobes = 3
    sinc_ovs = 10
    sincv = calc_sinc_vec(n_sinc_lobes, sinc_ovs)
    scat[100, 100] = 1
    srg[100, 100] = 50.5

    chan_profile_numba(srg.flatten(), scat.flatten(), binsize, min_sr,
                       sincv, n_sinc_lobes, sinc_ovs,
                       out1f)
    chan_profile_numba(srg, scat, binsize, min_sr,
                       sincv, n_sinc_lobes, sinc_ovs,
                       out1, rg_only=True)
    #plt.plot(np.linspace(0, 110, 110), np.real(out1))
    #scat = np.random.randn(N) + 1j * np.random.randn(N)
