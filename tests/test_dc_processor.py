import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from oceansar import ocs_io
from oceansar.constants import c
from oceansar.radarsim.dc_processor import dc_process, main


class DCProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'proc.nc'
        fs = 1000.
        slc = np.zeros((2, 2, 128, 64), complex)
        for channel in range(2):
            for pol in range(2):
                frequency = 40.+80*channel+30*pol
                slc[channel, pol] = np.exp(2j*np.pi*frequency*np.arange(128)[:, None]/fs)
        f = ocs_io.ProcFile(str(self.source), 'w', slc.shape)
        f.set('slc*', slc)
        for key, value in {'prf':fs, 'inc_angle':30., 'v_ground':5000.,
                           'rg_sampling':c/(2*5*.5), 'az0':-100.,
                           'sr0':800000., 'f0':5e9, 'num_ch':2}.items():
            f.set(key, value)
        f.close()

    def test_channel_polarizations_and_roundtrip(self):
        output = self.root/'dc.nc'
        ds = dc_process(None, self.source, output, resolution=(100., 100.),
                        taper='hann', channel=1)
        np.testing.assert_allclose(ds.dc_mean.sel(estimator='cde'), [120., 150.], atol=1e-5)
        self.assertEqual(ds.polarization.values.tolist(), ['pol0', 'pol1'])
        self.assertEqual(ds.attrs['channel_index'], 1)
        np.testing.assert_allclose(ds.attrs['pixel_spacing_m'], [5., 5.])
        with xr.open_dataset(output, engine='h5netcdf') as saved:
            np.testing.assert_allclose(saved.dc, ds.dc)
            np.testing.assert_allclose(saved.radial_velocity.sel(estimator='cde'),
                                       np.broadcast_to(np.array([-120., -150.])[:, None, None] * c/(2*5e9),
                                                       saved.radial_velocity.sel(estimator='cde').shape), atol=1e-6)
            self.assertEqual(saved.radial_velocity.attrs['units'], 'm/s')
        self.assertTrue(np.all(np.array(ds.attrs['output_spacing_m']) <=
                               np.array(ds.attrs['effective_resolution_m'])/2+1e-10))
        self.assertGreater(float(ds.azimuth[0]), -100.)

    def test_config_and_cli(self):
        config = self.root/'simulation.cfg'
        config.write_text(f'[sim]\npath={self.root}\nproc_file=proc.nc\ndc_file=cli.nc\n'
                          '[mode]\npol=DP\n[dc]\nresolution=[100, 100]\n'
                          'taper=hann\nchannel=1\npolarization=vv\nlag=1\n')
        ds = dc_process(config, self.source)
        self.assertEqual(ds.polarization.values.tolist(), ['vv'])
        self.assertAlmostEqual(float(ds.dc_mean.sel(estimator='cde').item()), 150., places=5)
        self.assertEqual(main(['-c', str(config)]), 0)
        self.assertTrue((self.root/'cli.nc').exists())
        self.assertEqual(dc_process(config, self.source, taper='boxcar').attrs['local_taper'], 'boxcar')

    def test_all_channels_are_independent(self):
        output = self.root/'all_dc.nc'
        ds = dc_process(None, self.source, output, resolution=100., channel='all')
        self.assertEqual(ds.channel.values.tolist(), [0, 1])
        np.testing.assert_allclose(ds.dc_mean.sel(estimator='cde'), [[40., 70.], [120., 150.]], atol=1e-5)
        for channel in range(2):
            single = dc_process(None, self.source, resolution=100., channel=channel)
            np.testing.assert_allclose(ds.dc.sel(channel=channel), single.dc)
        with xr.open_dataset(output, engine='h5netcdf') as saved:
            self.assertEqual(saved.attrs['channel_index'], 'all')
            np.testing.assert_allclose(saved.dc_mean, ds.dc_mean)
        from unittest.mock import patch
        from oceansar.radarsim.dc_processor import plot_dc
        with patch('matplotlib.pyplot.show') as show:
            plot_dc(ds, self.root/'all_plots', show=True)
        self.assertEqual(show.call_count, 16)
        self.assertTrue((self.root/'all_plots'/'dc_ch1_pol0_cde.png').exists())
        from oceansar.radarsim.doppler_plots import plot_dc_summary
        with patch('matplotlib.pyplot.show') as show:
            average = plot_dc_summary(ds, self.root/'summary', show=True)
        self.assertEqual(show.call_count, 2)  # One combined figure per polarization.
        np.testing.assert_allclose(average.dc_mean.sel(estimator='cde'), [80., 110.], atol=1e-5)
        np.testing.assert_allclose(average.dc, ds.dc.mean('channel'))
        self.assertTrue((self.root/'summary'/'dc_channel_summary_pol0.png').exists())

    def test_fill_mask_and_invalid_inputs(self):
        with Dataset(self.source, 'a') as file:
            file['slc_r'][0, 0, 4, 4] = np.ma.masked
        ds = dc_process(None, self.source, resolution=100., polarization=0)
        self.assertEqual(int(ds.mean_valid_pairs.item()), 127*64-2)
        for kwargs in ({'channel':3, 'resolution':100.}, {'resolution':None},
                       {'resolution':100., 'polarization':8}):
            with self.assertRaises(ValueError):
                dc_process(None, self.source, **kwargs)
        with self.assertRaises(ValueError):
            dc_process(None, self.source, self.source, resolution=100.)
        detected = self.root/'detected.nc'
        with Dataset(detected, 'w') as file:
            file.createDimension('x', 1)
            file.createVariable('ml_intensity', 'f4', ('x',))
        with self.assertRaises(ValueError):
            dc_process(None, detected, resolution=100.)

    def test_configured_plot_directory_and_inline_display(self):
        from unittest.mock import patch
        config = self.root/'plots.cfg'
        config.write_text('[dc]\nresolution=[100, 100]\nplot_save=True\n'
                          'plot_path=dc_plots\nsamples_per_resolution=4\n')
        with patch('matplotlib.pyplot.show') as show:
            ds = dc_process(config, self.source, self.root/'dc.nc', plot_show=True)
        self.assertEqual(ds.attrs['samples_per_resolution'], [4., 4.])
        self.assertEqual(show.call_count, 8)
        plots = list((self.root/'dc_plots').glob('*.png'))
        self.assertEqual(len(plots), 8)
        self.assertTrue(all(path.stat().st_size > 1000 for path in plots))
        self.assertFalse(list(self.root.glob('*.png')))


if __name__ == '__main__':
    unittest.main()
