import importlib
import unittest

import numpy as np

from oceansar.l2_processing_core.cross_spectrum import (
    outlier_removal, quick_xspec, welch_xspec,
)


HOST = welch_xspec.__module__.split('.')[0]


class CrossSpectrumTests(unittest.TestCase):
    def images(self, shape=(32, 48)):
        rng = np.random.default_rng(71)
        return tuple(1 + rng.random(shape) for _ in range(2))

    def test_frequency_phase_sign_and_amplitude(self):
        y, x = np.indices((32, 48))
        angle = 2*np.pi*(4*y/32 + 5*x/48)
        fwd, bwd = 1+.2*np.cos(angle), 1+.2*np.cos(angle+.6)
        kx, ky, spectrum, tiles = welch_xspec(
            fwd, bwd, pixel_spacing=(3., 7.), window_size=1e6,
            remove_outliers=False, win_func=np.ones)
        index = (np.argmin(abs(ky - 4/(32*3))), np.argmin(abs(kx - 5/(48*7))))
        self.assertAlmostEqual(float(np.angle(spectrum[index])), -.6, places=6)
        self.assertAlmostEqual(float(abs(spectrum[index])), fwd.size*.2**2/4, places=5)
        self.assertEqual(tiles.shape, (1, 1, 32, 48))
        reverse = welch_xspec(bwd, fwd, pixel_spacing=(3., 7.), window_size=1e6,
                              remove_outliers=False, win_func=np.ones)[2]
        np.testing.assert_allclose(reverse, spectrum.conj(), atol=1e-6)

    def test_window_energy_normalization_and_parseval(self):
        image = self.images()[0]
        spectrum = welch_xspec(image, image, pixel_spacing=(2., 5.),
                               window_size=1e6, remove_outliers=False)[2]
        window = np.hamming(32)[:, None] * np.hamming(48)[None, :]
        expected_variance = np.sum(((image-image.mean())/image.mean()*window)**2) / np.sum(window**2)
        self.assertAlmostEqual(float(spectrum.real.mean()), expected_variance, places=7)
        np.testing.assert_allclose(spectrum.imag, 0, atol=1e-7)

    def test_anisotropic_coordinates_and_full_tile_layout(self):
        fwd, bwd = self.images((23, 37))
        kx, ky, average, tiles = welch_xspec(
            fwd, bwd, pixel_spacing=(3., 7.), window_size=(36., 56.),
            remove_outliers=False)
        self.assertEqual(tiles.shape, (2, 8, 12, 8))
        self.assertEqual(average.shape, (12, 8))
        np.testing.assert_array_equal(kx, np.fft.fftshift(np.fft.fftfreq(8, 7.)))
        np.testing.assert_array_equal(ky, np.fft.fftshift(np.fft.fftfreq(12, 3.)))

    def test_scalar_support_and_small_image(self):
        fwd, bwd = self.images((7, 11))
        scalar = welch_xspec(fwd, bwd, pixel_spacing=5., window_size=2000., remove_outliers=False)
        pair = welch_xspec(fwd, bwd, pixel_spacing=5., window_size=(2000., 2000.), remove_outliers=False)
        for a, b in zip(scalar, pair):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(scalar[3].shape, (1, 1, 7, 11))

    def test_bright_tile_rejection_uses_original_tile_means(self):
        image = self.images((24, 24))[0]
        image[-8:, -8:] *= 100
        _, _, average, tiles = welch_xspec(
            image, image, pixel_spacing=1., window_size=8., remove_outliers=False,
            tile_normalization=False, tile_outlier_threshold=1.)
        levels = np.array([[image[y:y+8, x:x+8].mean() for x in range(0, 17, 4)]
                           for y in range(0, 17, 4)])
        keep = levels <= levels.mean() + levels.std()
        self.assertTrue(keep.any() and (~keep).any())
        np.testing.assert_array_equal(average, tiles[keep].mean(axis=0))
        self.assertFalse(np.allclose(average, tiles.mean(axis=(0, 1))))

    def test_zero_normalization_tiles_are_excluded(self):
        image = np.zeros((12, 12))
        image[-4:, -4:] = self.images((4, 4))[0]
        _, _, average, tiles = welch_xspec(
            image, image, pixel_spacing=1., window_size=4., remove_outliers=False)
        self.assertTrue(np.isnan(tiles).any())
        self.assertTrue(np.isfinite(average).all())
        with self.assertRaisesRegex(ValueError, 'No usable'):
            welch_xspec(image*0, image*0, pixel_spacing=1., remove_outliers=False)

    def test_invalid_inputs_are_rejected(self):
        image = np.ones((16, 16))
        for bad in (image.astype(complex), -image, image*np.nan, image[:, :8], image[:1]):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                welch_xspec(image, bad, pixel_spacing=5.)
        for options in ({'pixel_spacing': 0}, {'pixel_spacing': (1, np.nan)},
                        {'window_size': -1}, {'tile_outlier_threshold': -1},
                        {'win_func': lambda n: np.zeros(n)},
                        {'win_func': lambda n: -np.ones(n)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                welch_xspec(image, image, **(dict(pixel_spacing=5.) | options))

    def test_outlier_removal_and_zero_reference(self):
        image = np.ones((16, 16))
        image[8, 8] = 1000.
        filtered = outlier_removal(image, thresh=1., lp_scale=25., pixel_spacing=5.)
        self.assertEqual(filtered[8, 8], 1e-6)
        self.assertEqual(image[8, 8], 1000.)
        zeros = outlier_removal(image*0, second=True)
        np.testing.assert_array_equal(zeros, 0.)

    def test_quick_spectrum_scaling_and_coordinates(self):
        fwd, bwd = self.images((18, 23))
        options = dict(avg=(1, 1), remove_outliers=False)
        kx, ky, spectrum = quick_xspec(fwd, bwd, pixel_spacing=(2., 5.), **options)
        self.assertEqual(spectrum.shape, (32, 32))
        scaled = quick_xspec(3*fwd, 3*bwd, pixel_spacing=(4., 10.), **options)
        np.testing.assert_array_equal(scaled[0], kx/2)
        np.testing.assert_array_equal(scaled[1], ky/2)
        np.testing.assert_allclose(scaled[2], spectrum*9, rtol=3e-5, atol=1e-4)
        reverse = quick_xspec(bwd, fwd, pixel_spacing=(2., 5.), **options)[2]
        np.testing.assert_allclose(reverse, spectrum.conj(), atol=1e-6)

    def test_quick_spectral_smoothing_and_decimation(self):
        fwd, bwd = self.images((18, 23))
        kx, ky, spectrum = quick_xspec(fwd, bwd, pixel_spacing=5.)
        self.assertEqual(spectrum.shape, (11, 11))
        self.assertEqual(spectrum.shape, (len(ky), len(kx)))
        self.assertTrue(np.isfinite(spectrum).all())
        with self.assertRaises(ValueError):
            quick_xspec(fwd, bwd, pixel_spacing=5., avg=(0, 1))

    def test_project_welch_adapter_preserves_input_convention(self):
        fwd, bwd = self.images()
        module = ('radarsim.L2_wavespectrum' if HOST == 'oceansar' else 'processing.xspectra')
        adapter = importlib.import_module(f'{HOST}.{module}').welch_xspec
        inputs = (fwd, bwd) if HOST == 'oceansar' else (np.sqrt(fwd)*1j, np.sqrt(bwd)*np.exp(.7j))
        options = dict(window_size=(24., 40.), remove_outliers=False)
        actual = adapter(*inputs, {'az_spacing': 2., 'grg_spacing': 5.}, **options)
        expected = welch_xspec(fwd, bwd, pixel_spacing=(2., 5.), **options)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)

    def test_project_outlier_adapter(self):
        module = ('radarsim.L2_wavespectrum' if HOST == 'oceansar' else 'processing.preprocessing')
        adapter = importlib.import_module(f'{HOST}.{module}').outlier_removal
        image = self.images()[0]
        image[5, 5] = 1000
        options = dict(second=True, lp_scale=25., thresh=1.)
        np.testing.assert_array_equal(adapter(image, **options), outlier_removal(image, **options))

    @unittest.skipUnless(HOST == 's1d_ewv', 'S1-specific complex-input adapter')
    def test_s1_quick_adapter(self):
        adapter = importlib.import_module('s1d_ewv.processing.xspectra').quick_xspec
        fwd, bwd = self.images()
        options = dict(pixel_spacing=(3., 8.), remove_outliers=False)
        actual = adapter(np.sqrt(fwd)*1j, np.sqrt(bwd), **options)
        expected = quick_xspec(fwd, bwd, **options)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
