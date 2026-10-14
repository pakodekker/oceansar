import unittest

import numpy as np

from oceansar.radarsim.dc_estimator import estimate_dc


class DopplerTests(unittest.TestCase):
    def estimate(self, slc, **kwargs):
        return estimate_dc(slc, fs_az=1000., pixel_spacing=(2., 3.),
                           resolution=(32., 36.), **kwargs)

    def test_tones_and_grid(self):
        for frequency in (123., -237., 500., 620.):
            slc = np.exp(2j*np.pi*frequency*np.arange(64)[:, None]/1000)*np.ones((1, 24))
            if frequency == 500.:
                # Exact Nyquist tone; tiny phase noise can select either alias.
                slc = ((-1.)**np.arange(64)[:, None]*np.ones((1, 24))).astype(complex)
            ds = self.estimate(slc, methods='cde')
            expected = (frequency+500) % 1000-500
            np.testing.assert_allclose(ds.dc, expected, atol=1e-10)
            np.testing.assert_allclose(ds.dc_mean, expected, atol=1e-10)
            np.testing.assert_allclose(np.diff(ds.azimuth), 16.)
            np.testing.assert_allclose(np.diff(ds.range), 18.)
            self.assertEqual(int(ds.mean_valid_pairs), 63*24)

    def test_denser_sampling_keeps_resolution_and_shared_estimates(self):
        rng = np.random.default_rng(12)
        slc = rng.normal(size=(64, 48)) + 1j*rng.normal(size=(64, 48))
        default = self.estimate(slc)
        dense = self.estimate(slc, samples_per_resolution=4)
        np.testing.assert_equal(default.attrs['window_samples'], dense.attrs['window_samples'])
        np.testing.assert_equal(default.attrs['effective_resolution_m'], dense.attrs['effective_resolution_m'])
        np.testing.assert_allclose(dense.dc.sel(azimuth=default.azimuth, range=default.range), default.dc)
        np.testing.assert_array_equal(dense.dc_mean.values, default.dc_mean.values)
        self.assertGreater(dense.sizes['azimuth'], default.sizes['azimuth'])
        self.assertGreater(dense.sizes['range'], default.sizes['range'])
        limited = self.estimate(slc, samples_per_resolution=100)
        np.testing.assert_equal(limited.attrs['output_stride_samples'], [1, 1])
        for invalid in (0, -1, .5, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.estimate(slc, samples_per_resolution=invalid)

    def test_sde_gaussian_model(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(size=(512, 128))+1j*rng.normal(size=(512, 128))
        slc = noise.copy()
        coefficient = .8*np.exp(2j*np.pi*137/1000)
        for row in range(1, len(slc)):
            slc[row] += coefficient*slc[row-1]
        ds = self.estimate(slc)
        np.testing.assert_allclose(ds.dc_mean, 137., atol=2.)
        scaled = self.estimate(17*slc, methods='sde')
        np.testing.assert_allclose(scaled.dc.sel(estimator='sde'), ds.dc.sel(estimator='sde'))

    def test_mask_segments_and_pooled_mean(self):
        rng = np.random.default_rng(4)
        slc = rng.normal(size=(64, 24))+1j*rng.normal(size=(64, 24))
        mask = np.ones(slc.shape, bool)
        mask[10:13] = False
        slc[20, 2] = np.nan
        segments = np.repeat([0, 1], 32)
        ds = self.estimate(slc, lag=2, valid_mask=mask, segment_ids=segments)
        valid = mask & np.isfinite(slc)
        pairs = valid[2:] & valid[:-2]
        pairs &= (segments[2:] == segments[:-2])[:, None]
        products = slc[2:][pairs]*slc[:-2][pairs].conj()
        expected = np.angle(products.sum())*1000/(4*np.pi)
        self.assertEqual(int(ds.mean_valid_pairs), len(products))
        self.assertAlmostEqual(float(ds.dc_mean.sel(estimator='cde')), expected)

    def test_invalid_support_and_rounding(self):
        ds = self.estimate(np.zeros((64, 24), complex), nodata=0)
        self.assertTrue(np.isnan(ds.dc).all())
        self.assertEqual(int(ds.mean_valid_pairs), 0)
        ds = estimate_dc(np.ones((20, 20), complex), fs_az=1000,
                         pixel_spacing=(3., 4.), resolution=(19., 25.))
        self.assertEqual(ds.attrs['resolution_m'], [21., 28.])
        self.assertEqual(ds.attrs['window_samples'], [8, 7])
        self.assertEqual(ds.attrs['output_spacing_m'], [9., 12.])
        with self.assertRaises(ValueError):
            self.estimate(np.ones((4, 4), complex))
        with self.assertRaises(ValueError):
            self.estimate(np.ones((64, 24)))

    def test_weighted_correlations_with_mask(self):
        rng = np.random.default_rng(8)
        slc = rng.normal(size=(8, 6))+1j*rng.normal(size=(8, 6))
        mask = np.ones(slc.shape, bool)
        mask[3, 2] = False
        weights = np.arange(42, dtype=float).reshape(7, 6)
        ds = estimate_dc(slc, fs_az=1000., pixel_spacing=1., resolution=(2., 2.),
                         taper=weights, valid_mask=mask, window_samples=(8, 6))
        a, b = slc[1:], slc[:-1]
        keep = mask[1:] & mask[:-1] & (weights > 0)
        w = weights[keep]/weights.max()
        a, b = a[keep], b[keep]
        cde = np.average(a*b.conj(), weights=w)
        component = lambda a, b: np.sin(np.pi/2 * np.average(
            np.where(a >= 0, 1., -1.)*np.where(b >= 0, 1., -1.), weights=w))
        sde = .5*(component(a.real, b.real)+component(a.imag, b.imag)) + .5j*(
            component(a.imag, b.real)-component(a.real, b.imag))
        np.testing.assert_allclose(ds.correlation_real[:, 0, 0], [cde.real, sde.real])
        np.testing.assert_allclose(ds.correlation_imag[:, 0, 0], [cde.imag, sde.imag])
        np.testing.assert_allclose(ds.dc[:, 0, 0], np.angle([cde, sde])*1000/(2*np.pi))
        self.assertAlmostEqual(float(ds.effective_pairs[0, 0]), w.sum()**2/(w*w).sum())
        self.assertEqual(int(ds.valid_pairs[0, 0]), len(w))

    def test_hann_tone_and_unchanged_image_mean(self):
        slc = np.exp(2j*np.pi*123*np.arange(64)[:, None]/1000)*np.ones((1, 24))
        rectangular, tapered = self.estimate(slc), self.estimate(slc, taper='hann')
        np.testing.assert_allclose(tapered.dc.sel(estimator='cde'), 123., atol=1e-10)
        np.testing.assert_array_equal(tapered.dc_mean, rectangular.dc_mean)
        self.assertTrue(np.all(np.array(tapered.attrs['window_samples']) > rectangular.attrs['window_samples']))
        self.assertTrue(np.all(np.array(tapered.attrs['effective_resolution_m']) >= [32., 36.]))
        self.assertTrue(np.all(np.array(tapered.attrs['output_spacing_m']) <=
                               np.array(tapered.attrs['effective_resolution_m'])/2+1e-12))
        boxcar = self.estimate(slc, taper='boxcar')
        np.testing.assert_allclose(boxcar.dc, rectangular.dc)
        for invalid in (np.zeros((15, 12)), -np.ones((15, 12)),
                        np.full((15, 12), np.nan), np.ones((16, 12))):
            with self.assertRaises(ValueError):
                self.estimate(slc, taper=invalid)

    def test_discrete_hann_sizing_and_lag_support(self):
        from scipy.signal.windows import hann
        for lag in (1, 3):
            ds = estimate_dc(np.ones((200, 200), complex), fs_az=1000.,
                             pixel_spacing=(2., 3.), resolution=(100., 120.),
                             taper='hann', lag=lag)
            for axis, (dx, target) in enumerate(((2., 100.), (3., 120.))):
                n = ds.attrs['window_samples'][axis]-(lag if axis == 0 else 0)
                def effective(n):
                    w = hann(n)
                    x = np.arange(n)*dx
                    centre = np.average(x, weights=w)
                    return np.sqrt(12*np.average((x-centre)**2, weights=w)+dx**2)
                self.assertAlmostEqual(ds.attrs['effective_resolution_m'][axis], effective(n))
                self.assertGreaterEqual(effective(n), target)
                self.assertLess(effective(n-1), target)
                self.assertAlmostEqual(n*dx/target, 1.597, delta=.07)
                self.assertEqual(ds.attrs['output_stride_samples'][axis],
                                 int(np.floor(effective(n)/(2*dx))))
        with self.assertRaises(ValueError):
            estimate_dc(np.ones((20, 20), complex), fs_az=1000., pixel_spacing=1.,
                        resolution=15., taper='hann')

    def test_masked_spatial_moments_and_explicit_custom_taper(self):
        slc = np.ones((10, 8), complex)
        mask = np.ones(slc.shape, bool)
        mask[:, :4] = False
        ds = estimate_dc(slc, fs_az=1000., pixel_spacing=1., resolution=(4., 4.),
                         window_samples=(10, 8), taper=np.ones((9, 8)), valid_mask=mask)
        self.assertAlmostEqual(float(ds.valid_resolution_range[0, 0]), 4.)
        self.assertAlmostEqual(float(ds.averaging_centre_shift_range[0, 0]), 2.)
        self.assertAlmostEqual(float(ds.range[0]), 3.5)
        self.assertAlmostEqual(float(ds.valid_resolution_azimuth[0, 0]), 9.)
        with self.assertRaises(ValueError):
            estimate_dc(slc, fs_az=1000., pixel_spacing=1., resolution=8.,
                        window_samples=(5, 5))


if __name__ == '__main__':
    unittest.main()
