import unittest

import numpy as np
from scipy.signal import fftconvolve

from oceansar.closure import randomscat_ts


class RandomScattererTests(unittest.TestCase):
    def test_seed_and_rng_reproducibility(self):
        for make_rng in (lambda: {'seed': 17},
                         lambda: {'rng': np.random.default_rng(17)},
                         lambda: {'rng': np.random.RandomState(17)}):
            a = randomscat_ts(.25, (3, 16, 12), 32, **make_rng())
            b = randomscat_ts(.25, (3, 16, 12), 32, **make_rng())
            for time in (0., .25, -.5):
                np.testing.assert_array_equal(a.scats(time), b.scats(time))
        old_state = np.random.get_state()
        try:
            np.random.seed(21)
            a = randomscat_ts(.25, 100, 32)
            np.random.seed(21)
            b = randomscat_ts(.25, 100, 32)
            np.testing.assert_array_equal(a.scats(0), b.scats(0))
        finally:
            np.random.set_state(old_state)
        a = randomscat_ts(.25, 100, 32, seed=1)
        b = randomscat_ts(.25, 100, 32, seed=2)
        self.assertFalse(np.array_equal(a.scats(0), b.scats(0)))
        with self.assertRaises(ValueError):
            randomscat_ts(.25, 100, 32, seed=1, rng=np.random.default_rng(1))

    def test_full_rounded_pool_and_compact_storage(self):
        pool = randomscat_ts(.25, 20000, 32, N=17, seed=12)
        self.assertGreater(pool.N, 17)
        self.assertEqual(pool._packed_indices.dtype, np.dtype('uint32'))
        self.assertEqual(pool._packed_indices.nbytes, 4*20000)
        self.assertEqual(pool.ind0.min(), 0)
        self.assertEqual(pool.ind0.max(), pool.N-1)
        self.assertAlmostEqual(float(np.mean(abs(pool.s)**2)), 1.)

    def test_fixed_phases_preserve_per_cell_temporal_products(self):
        pool = randomscat_ts(.25, 10000, 32, seed=14)
        x0 = pool.scats(0.)
        unrotated0 = pool.s[pool.ind0]
        # Rotate by one of the four phases, without altering any amplitudes.
        np.testing.assert_allclose(abs(x0), abs(unrotated0), rtol=1e-15)
        phases = x0 / unrotated0
        self.assertTrue(np.all(np.min(abs(phases[:, None] -
                                          np.array([1, 1j, -1, -1j])), axis=1) < 1e-14))
        for step in (1, 8, -5, pool.N+4):
            unrotated = pool.s[(pool.ind0.astype(np.int64)+step) % pool.N]
            np.testing.assert_allclose(
                pool.scats(step/pool.Fs) * x0.conj(),
                unrotated * unrotated0.conj(), rtol=1e-14, atol=1e-14,
            )

    def test_spatial_randomization_even_for_a_constant_shared_sequence(self):
        pool = randomscat_ts(.25, 262144, 32, N=1, seed=7)
        x = pool.scats(0.)
        self.assertLess(abs(x.mean()), .01)
        self.assertLess(abs(np.mean(x*x)), .01)
        self.assertLess(abs(np.mean(x[::2]*x[1::2].conj())), .01)
        np.testing.assert_array_equal(x, pool.scats(100.))

    def test_gaussian_temporal_correlation(self):
        tau, fs = .25, 32
        pool = randomscat_ts(tau, 262144, fs, N=65536, seed=123)
        x = pool.scats(0.)
        power = np.mean(abs(x)**2)
        self.assertAlmostEqual(float(power), 1., delta=.02)
        for step in (2, 4, 8, 16):
            correlation = np.mean(pool.scats(step/fs)*x.conj()) / power
            expected = np.exp(-(step/fs/tau)**2)
            self.assertLess(abs(correlation-expected), .025)

    def test_slices_buffers_scalars_and_stateless_time_access(self):
        pool = randomscat_ts(.25, (3, 8, 10), 32, seed=4)
        expected = pool.scats(.25)
        for index in (1, (1, slice(None, None, -2), slice(1, None, 2)),
                      (slice(None), [1, 3], slice(None)),
                      (slice(0, 0),), (1, 2, 3)):
            selected = expected[index]
            np.testing.assert_array_equal(pool.scats(.25, index), selected)
            out = np.empty(np.shape(selected), complex)
            self.assertIs(pool.scats(.25, index, out=out), out)
            np.testing.assert_array_equal(out, selected)
        out = np.empty((3, 8, 20), complex)[:, :, ::2]
        self.assertIs(pool.scats(.25, out=out), out)
        np.testing.assert_array_equal(out, expected)
        out64 = np.empty(expected.shape, np.complex64)
        pool.scats(.25, out=out64)
        np.testing.assert_array_equal(out64, expected.astype(np.complex64))
        self.assertTrue(np.isscalar(pool.scats(.25, (1, 2, 3))))
        pool.scats(-3.)
        pool.scats(7.)
        np.testing.assert_array_equal(pool.scats(.25), expected)
        np.testing.assert_array_equal(pool.scats(.25+.2/pool.Fs), expected)
        np.testing.assert_array_equal(pool.scats(.25+pool.N/pool.Fs), expected)
        np.testing.assert_array_equal(pool.scats(-1/pool.Fs),
                                      pool.scats((pool.N-1)/pool.Fs))
        readonly = np.empty(expected.shape, complex)
        readonly.flags.writeable = False
        for invalid in (np.empty((2, 3), complex), np.empty(expected.shape), readonly):
            with self.assertRaises(ValueError):
                pool.scats(0., out=invalid)

    def test_no_coherent_dc_spike_after_geometric_phase_history(self):
        # Frozen, flat scene: include the raw geometric chirp and antenna
        # weighting before looking for a coherent line in the complex raw FFT.
        fs, wavelength, slant_range = 1378., .238403545, 785500.
        effective_velocity, orbital_velocity = 7200.086, 7590.981
        chirp_rate = 2*effective_velocity**2/(wavelength*slant_range)
        scene_size, burst_size = 2019, 1653
        pool = randomscat_ts(.326584, (scene_size, 128), fs/54, seed=2026)
        half_support = (scene_size+burst_size)//2
        times = np.arange(-half_support, half_support+1)/fs
        sine = -effective_velocity**2*times/(orbital_velocity*slant_range)
        kernel = (np.exp(-1j*np.pi*chirp_rate*times**2)
                  * np.sinc(11/wavelength*sine) * np.sinc(2.2/wavelength*sine))
        raw = fftconvolve(pool.scats(0.), kernel[:, None], axes=0)
        start = (len(raw)-burst_size)//2
        raw = raw[start:start+burst_size]
        spectrum = np.mean(abs(np.fft.fft(raw, axis=0))**2, axis=1)
        neighbors = spectrum[np.r_[1:13, -12:0]]
        self.assertLess(spectrum[0] / np.median(neighbors), 2.)


if __name__ == '__main__':
    unittest.main()
