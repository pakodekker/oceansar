from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from oceansar.surfaces import OceanSurface


class OceanStateTests(unittest.TestCase):
    def make_surface(self, depth, swell):
        surface = OceanSurface()
        surface.init(512., 512., 8., 8., 'auto', 'elfouhaily', 'elfouhaily',
                     .7, 500e3, 5., .1, .3, compute=['D', 'Diff', 'V', 'hMTF'],
                     depth=depth, swell_enable=swell, swell_ampl=.5,
                     swell_dir=.3, swell_wl=80., choppy_enable=True,
                     workers=1, rng=np.random.default_rng(12))
        return surface

    def test_complex_storage_and_propagated_surface_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ocean.nc'
            for depth, swell in ((None, False), (30., True)):
                with self.subTest(depth=depth, swell=swell):
                    original = self.make_surface(depth, swell)
                    original.save(path)
                    with xr.open_dataset(path, engine='h5netcdf') as stored:
                        self.assertEqual(stored.wave_coefs.dtype, np.dtype('complex64'))
                        self.assertEqual(stored.kx.dims, ('x_dim',))
                        self.assertEqual(stored.ky.dims, ('y_dim',))
                        self.assertNotIn('wave_coefs_r', stored)
                        np.testing.assert_array_equal(stored.wave_coefs, original.wave_coefs)
                        payload = sum(v.nbytes for v in stored.variables.values())
                        self.assertLess(payload, 9*original.Nx*original.Ny)
                    loaded = OceanSurface()
                    loaded.load(path, compute=['D', 'Diff', 'V', 'hMTF'])
                    self.assertEqual(loaded.depth, depth)
                    np.testing.assert_array_equal(loaded.kx, original.kx)
                    np.testing.assert_array_equal(loaded.ky, original.ky)
                    for time in (0., .5, 2.):
                        loaded.t = original.t = time
                        for name in ('Dx', 'Dy', 'Dz', 'Diffx', 'Diffy', 'Vx', 'Vy', 'Vz', 'hMTF'):
                            np.testing.assert_allclose(getattr(loaded, name), getattr(original, name),
                                                       rtol=2e-6, atol=2e-7)

    def test_higher_precision_coefficients_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ocean.nc'
            original = self.make_surface(None, False)
            original.wave_coefs = original.wave_coefs.astype(np.complex128)
            original.wave_coefs[1, 1] += 1e-10 + 2e-10j
            original.save(path)
            loaded = OceanSurface()
            loaded.load(path)
            self.assertEqual(loaded.wave_coefs.dtype, np.dtype('complex128'))
            np.testing.assert_array_equal(loaded.wave_coefs, original.wave_coefs)


if __name__ == '__main__':
    unittest.main()
