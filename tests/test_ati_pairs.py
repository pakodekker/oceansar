import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from oceansar.constants import c
from oceansar.radarsim.ati_pairs import estimate_ati_pairs
from oceansar.radarsim.doppler_plots import plot_ati


class ATIPairTests(unittest.TestCase):
    def fixture(self, baselines=None):
        baselines = np.array([0., 2., 5., 9., 14.] if baselines is None else baselines)
        nch, npol = len(baselines), 2
        lut = np.full((nch, npol, nch, npol), -1)
        correlations = []
        # Same storage order as the InSAR extended covariance matrix, including cross-pol entries.
        for first in range(nch*npol):
            for last in range(first+1, nch*npol):
                i, p = divmod(first, npol)
                j, q = divmod(last, npol)
                lut[i, p, j, q] = len(correlations)
                frequency = 20. + 15*p + 3*i + 2*(j-i)
                tau = (baselines[j]-baselines[i]) / (2*7500.)
                correlations.append(np.full((4, 3), .9*np.exp(2j*np.pi*frequency*tau)))
        return np.array(correlations), lut, baselines

    def estimate(self, correlations, lut, baselines):
        return estimate_ati_pairs(correlations, lut, baselines, 7500., 5e9,
                                  ['hh', 'vv'], np.arange(4), np.arange(3))

    def test_all_pairs_conversion_and_lag_averages(self):
        correlations, lut, baselines = self.fixture()
        ds = self.estimate(correlations, lut, baselines)
        self.assertEqual(ds.sizes['pair'], 10)
        np.testing.assert_array_equal(ds.lag_pair_count.values, [4, 3, 2, 1])
        for i in range(5):
            for j in range(i+1, 5):
                pair = ds.sel(pair=f'{i}-{j}')
                expected = np.array([20., 35.]) + 3*i + 2*(j-i)
                np.testing.assert_allclose(pair.pair_dc_mean, expected, atol=1e-10)
                np.testing.assert_allclose(pair.pair_radial_velocity_mean, -c*expected/(2*5e9))
                self.assertAlmostEqual(float(pair.time_separation_s), (baselines[j]-baselines[i])/15000)
        for lag in range(1, 5):
            expected = np.array([20., 35.]) + 3*np.mean(np.arange(5-lag)) + 2*lag
            np.testing.assert_allclose(ds.lag_dc_mean.sel(lag=lag), expected, atol=1e-10)
        np.testing.assert_allclose(ds.dc, ds.pair_dc.sel(pair='0-1'))
        # Longest lag contains exactly the first/last-antenna estimate.
        np.testing.assert_allclose(ds.lag_dc.sel(lag=4), ds.pair_dc.sel(pair='0-4'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'all_pairs.nc'
            ds.to_netcdf(path, engine='h5netcdf')
            with xr.open_dataset(path, engine='h5netcdf') as saved:
                np.testing.assert_allclose(saved.lag_dc, ds.lag_dc)
                self.assertEqual(saved.pair.values.tolist(), ds.pair.values.tolist())
            with patch('matplotlib.pyplot.show') as show:
                plot_ati(ds, directory, show=True)
            self.assertEqual(show.call_count, 18)  # 2 polarizations * (4 lags * 2 units + summary).
            self.assertTrue((Path(directory)/'ATI_lag_summary_vv.png').exists())

    def test_missing_pairs_and_invalid_pixels(self):
        correlations, lut, baselines = self.fixture()
        correlations[lut[0, 0, 1, 0], 0, 0] = np.nan
        ds = self.estimate(correlations, lut, baselines)
        self.assertEqual(int(ds.lag_valid_pair_count.sel(lag=1, polarization='hh')[0, 0]), 3)
        expected = np.mean([25., 28., 31.])  # Remaining adjacent-pair frequencies.
        self.assertAlmostEqual(float(ds.lag_dc.sel(lag=1, polarization='hh')[0, 0]), expected)
        lut[0, 0, 4, 0] = -1
        with self.assertRaisesRegex(ValueError, 'missing same-polarization pair 0-4'):
            self.estimate(correlations, lut, baselines)
        correlations, lut, baselines = self.fixture([0., 2., 2., 9., 14.])
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            self.estimate(correlations, lut, baselines)

    def test_spatial_std_and_pair_matrix_layout(self):
        correlations, lut, baselines = self.fixture()
        variation = np.arange(12).reshape(4, 3) * .5
        for i in range(5):
            for j in range(i+1, 5):
                tau = (baselines[j]-baselines[i])/15000.
                for pol in range(2):
                    correlations[lut[i, pol, j, pol]] *= np.exp(2j*np.pi*variation*tau)
        ds = self.estimate(correlations, lut, baselines)
        np.testing.assert_allclose(ds.pair_dc_std, variation.std(), atol=1e-10)
        np.testing.assert_allclose(ds.lag_dc_std, variation.std(), atol=1e-10)
        ds['intensity'] = (('channel', 'polarization', 'azimuth', 'range'),
                           np.broadcast_to((10**(np.arange(5)/10))[:, None, None, None], (5, 2, 4, 3)))
        ds = ds.assign_coords(channel=np.arange(5))
        from oceansar.radarsim.doppler_plots import plot_ati_matrix
        import matplotlib.pyplot as plt
        def check_layout():
            figure = plt.gcf()
            axes = {axis.get_label(): axis for axis in figure.axes}
            np.testing.assert_allclose(axes['antenna_0_4'].collections[0].get_array(),
                                       ds.pair_phase.sel(pair='0-4', polarization=current_pol))
            np.testing.assert_allclose(axes['antenna_4_0'].collections[0].get_array(),
                                       ds.pair_coherence.sel(pair='0-4', polarization=current_pol))
            np.testing.assert_allclose(axes['antenna_3_3'].collections[0].get_array(), 3.)
            orientations = [axis._colorbar.orientation for axis in figure.axes if hasattr(axis, '_colorbar')]
            self.assertEqual(sorted(orientations), ['horizontal', 'vertical'])
        # Inspect each polarization independently so matrix placement is verified against its data.
        with tempfile.TemporaryDirectory() as directory:
            for current_pol in ('hh', 'vv'):
                with patch('matplotlib.pyplot.show', side_effect=check_layout) as show:
                    plot_ati_matrix(ds.sel(polarization=[current_pol]), directory, show=True)
                self.assertEqual(show.call_count, 1)
                self.assertTrue((Path(directory)/f'ATI_pair_matrix_{current_pol}.png').exists())

    def test_matrix_and_estimate_maps_share_coordinate_bounds(self):
        ati = self.estimate(*self.fixture()).sel(polarization=['vv'])
        ati['intensity'] = (('channel', 'polarization', 'azimuth', 'range'), np.ones((5, 1, 4, 3)))
        ati = ati.assign_coords(channel=np.arange(5))
        values = np.ones((1, 2, 3, 2)) * 25.
        dc = xr.Dataset({
            'dc': (('polarization', 'estimator', 'azimuth', 'range'), values),
            'radial_velocity': (('polarization', 'estimator', 'azimuth', 'range'), -c*values/(2*5e9)),
        }, coords={'polarization': ['vv'], 'estimator': ['cde', 'sde'],
                   'azimuth': [.25, 1.5, 2.75], 'range': [0., 1.5]}, attrs={'channel_index': 0})
        dc['dc_mean'] = dc.dc.mean(('azimuth', 'range'))
        dc['radial_velocity_mean'] = dc.radial_velocity.mean(('azimuth', 'range'))
        from oceansar.radarsim.dc_processor import plot_dc
        from oceansar.radarsim.doppler_plots import (
            common_map_bounds, plot_ati_matrix, plot_dc_summary, plot_comparison)
        import matplotlib.pyplot as plt
        from matplotlib.collections import QuadMesh
        bounds = common_map_bounds(ati)
        self.assertEqual(bounds, {'azimuth': (0., 3.), 'range': (0., 2.)})
        checked = []
        def check_bounds():
            figure = plt.gcf()
            figure.canvas.draw()
            for axis in figure.axes:
                if axis.get_label() != '<colorbar>' and any(isinstance(mesh, QuadMesh) for mesh in axis.collections):
                    np.testing.assert_allclose(axis.get_ylim(), bounds['azimuth'])
                    np.testing.assert_allclose(axis.get_xlim(), bounds['range'])
                    box = axis.get_window_extent()
                    self.assertAlmostEqual(box.height/box.width, 3./2., places=7)
                    np.testing.assert_allclose(axis.get_yticks(), [0., 1.5, 3.])
                    checked.append(axis)
        with patch('matplotlib.pyplot.show', side_effect=check_bounds):
            plot_ati_matrix(ati, show=True, bounds=bounds)
            plot_ati(ati, show=True, bounds=bounds)
            plot_dc(dc, show=True, bounds=bounds)
            plot_dc_summary(dc, show=True, bounds=bounds)
            plot_comparison(dc, ati, show=True)
        self.assertGreater(len(checked), 40)

    def test_coherence_scale_padding_and_quantization(self):
        from oceansar.radarsim.doppler_plots import coherence_lower_limit
        for minimum, expected in ((.99, .7), (.95, .7), (.9, .6), (.8, .5), (.25, 0.), (.2, 0.)):
            self.assertEqual(coherence_lower_limit([minimum, 1., np.nan]), expected)
        self.assertEqual(coherence_lower_limit([np.nan]), 0.)


if __name__ == '__main__':
    unittest.main()
