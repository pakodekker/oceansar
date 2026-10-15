import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

from oceansar import ocs_io, utils
from oceansar.constants import c
from oceansar.radarsim.multilook import multilook_window
from oceansar.radarsim.insar_processor import insar_process
from oceansar.radarsim.ati_processor import ati_process


class MultilookTests(unittest.TestCase):
    def test_resolution_matches_actual_smoothing_kernel(self):
        spacing = np.array([3., 7.])
        for window in ('flat', 'hanning', 'hamming', 'bartlett', 'blackman'):
            counts, effective = multilook_window(spacing, window, resolution=[100, 150])
            self.assertTrue(np.all(effective >= [100, 150]))
            for n, dx, expected in zip(counts, spacing, effective):
                impulse = np.zeros(401)
                impulse[200] = 1
                kernel = utils.smooth(impulse, n, window=window)
                x = np.arange(kernel.size) * dx
                mean = np.dot(kernel, x) / kernel.sum()
                measured = np.sqrt(12 * np.dot(kernel, (x-mean)**2) / kernel.sum() + dx**2)
                self.assertAlmostEqual(measured, expected, places=6)
            _, previous = multilook_window(spacing, window, window_samples=counts-1)
            self.assertTrue(np.all(previous < [100, 150]))

    def test_geometry_and_legacy_controls(self):
        a, _ = multilook_window([2., 5.], 'hanning', resolution=100)
        b, _ = multilook_window([4., 10.], 'hanning', resolution=100)
        self.assertTrue(np.all(a > b))
        counts, _ = multilook_window([2., 5.], 'hanning', window_samples=[8, 12])
        np.testing.assert_equal(counts, [8, 12])
        with self.assertRaises(ValueError):
            multilook_window([2., 5.], resolution=100, image_shape=[10, 10])
        for resolution in (-1, np.nan):
            with self.assertRaises(ValueError):
                multilook_window([2., 5.], resolution=resolution)

    def test_insar_to_ati_resolution_metadata(self):
        class Surface:
            dx = dy = 5.
            Lx = Ly = 640.
            def load(self, *args, **kwargs):
                self.Vx = np.tile([[-.001, .001]], (128, 64))  # Zero mean and azimuth guard.
                self.Vz = np.zeros((128, 128))
                self.Dz = np.zeros((128, 128))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root/'sim.cfg'
            common = ('plot_save=True\nplot_path=plots\nplot_format=png\nplot_tex=False\n'
                      'plot_surface=False\nplot_proc_ampl=False\nplot_coh=False\nplot_coh_all=False\n')
            cfg.write_text('[mode]\npol=vv\nmode=stripmap\n[processing]\nreramp=False\n'
                           '[insar]\nresolution=[100, 100]\nml_win=hanning\nflatten=False\n' + common +
                           '[ati]\nml_win=flat\nplot_ati_phase=False\nplot_ati_phase_all=False\n'
                           'plot_vel_hist=False\nplot_vel=False\n' + common)
            rng = np.random.default_rng(7)
            # Constant azimuth signal: coregistration leaves the known ATI phase unchanged.
            base = rng.normal(size=(1, 1, 1, 128)) + 1j*rng.normal(size=(1, 1, 1, 128))
            slc = np.concatenate([np.broadcast_to(base, (1, 1, 256, 128)),
                                  np.broadcast_to(base * np.exp(.2j), (1, 1, 256, 128))])
            f = ocs_io.ProcFile(str(root/'slc.nc'), 'w', slc.shape)
            f.set('slc*', slc)
            for key, value in {'prf':1000., 'inc_angle':30., 'v_ground':5000.,
                               'rg_sampling':c/5, 'az0':-640., 'sr0':800000.,
                               'f0':5e9, 'num_ch':2, 'rg_bw':1e7,
                               'orbit_alt':693000., 'v_orbit':7500.,
                               'b_ati':[0., 2.], 'b_xti':[0., 0.]}.items():
                f.set(key, value)
            f.close()
            with patch('oceansar.radarsim.insar_processor.OceanSurface', Surface), \
                 patch('oceansar.radarsim.ati_processor.OceanSurface', Surface):
                insar_process(str(cfg), str(root/'slc.nc'), 'unused', str(root/'insar.nc'))
                ati = ati_process(str(cfg), str(root/'insar.nc'), 'unused', str(root/'ati.txt'))
                prefix, ati_config = cfg.read_text().split('[ati]\n')
                for flag in ('plot_surface', 'plot_proc_ampl', 'plot_coh',
                             'plot_ati_phase', 'plot_vel_hist', 'plot_vel'):
                    ati_config = ati_config.replace(f'{flag}=False', f'{flag}=True')
                cfg.write_text(prefix + '[ati]\n' + ati_config)
                import matplotlib.pyplot as plt
                shown_titles = []
                with patch('matplotlib.pyplot.show', side_effect=lambda:
                           shown_titles.append(plt.gcf()._suptitle.get_text() if plt.gcf()._suptitle
                                               else plt.gcf().axes[0].get_title())):
                    ati_process(str(cfg), str(root/'insar.nc'), 'unused',
                                str(root/'ati.txt'), plot_show=True)
                self.assertEqual(len(shown_titles), 10)
                self.assertTrue(any('phase/coherence/intensity matrix' in title for title in shown_titles))
                self.assertIn('Estimated velocity', shown_titles)
                self.assertTrue((root/'plots'/'ATI_pair_matrix_vv.png').exists())
                self.assertFalse((root/'plots'/'ATI_coh_vvvv.png').exists())
                self.assertTrue((root/'plots'/'ATI_radial_vel_hist.png').exists())
                self.assertEqual(plt.get_fignums(), [])
            with Dataset(root/'insar.nc') as saved:
                counts = np.array([saved['az_ml'][:], saved['rg_ml'][:]])
                self.assertTrue(np.all(saved.ml_effective_resolution_m >= 100))
                np.testing.assert_allclose(saved.ml_window_support_m, counts * 5)
                self.assertEqual(saved.ml_window, 'hanning')
                # ROI is 128x128; only complete 33x33 averaging windows survive.
                self.assertEqual(ati.sizes['azimuth'], 96)
                self.assertEqual(ati.sizes['range'], 96)
                np.testing.assert_allclose(ati.azimuth[[0, -1]], [-240, 235])
                np.testing.assert_allclose(ati.range[[0, -1]], [80, 555])
            expected_hz = .2 / (2*np.pi*(1/7500))
            np.testing.assert_allclose(ati.dc, expected_hz, rtol=1e-6)
            np.testing.assert_allclose(ati.radial_velocity, -c*expected_hz/(2*5e9), rtol=1e-6)
            from oceansar.radarsim.dc_processor import dc_process
            from oceansar.radarsim.doppler_plots import plot_comparison, plot_ati
            dc = dc_process(str(cfg), root/'slc.nc', resolution=100, taper='hann', plot=False)
            with patch('matplotlib.pyplot.show') as show:
                plot_ati(ati, root/'plots', show=True)
                plot_comparison(dc, ati, root/'comparison', show=True)
            self.assertEqual(show.call_count, 4)
            self.assertTrue((root/'comparison'/'dc_ati_vv.png').exists())
            self.assertTrue((root/'plots'/'ATI_doppler_vvvv.png').exists())
            import xarray as xr
            with xr.open_dataset(root/'ati_maps.nc', engine='h5netcdf') as saved:
                np.testing.assert_allclose(saved.dc, ati.dc)
            self.assertIn('Inherited InSAR Multilooking (hanning', (root/'ati.txt').read_text())
            self.assertIn('lag_dc_std', (root/'ati.txt').read_text())
            self.assertIn('pair_dc_std', (root/'ati.txt').read_text())
            with np.load(root/'ati_stats.npz') as stats:
                np.testing.assert_equal(stats['ml_window_samples'], counts)
                self.assertEqual(str(stats['ml_window']), 'hanning')
                np.testing.assert_allclose(stats['lag_doppler_std_hz'], ati.lag_dc_std)


if __name__ == '__main__':
    unittest.main()
