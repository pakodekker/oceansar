import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

from oceansar import ocs_io
from oceansar.radarsim.sar_raw import sar_raw
from oceansar.surfaces import OceanSurface


class FrozenRawTests(unittest.TestCase):
    def test_frozen_surface_is_initialized_once_for_new_and_reused_ocean(self):
        updates = []

        class CheckedSurface(OceanSurface):
            def __getattribute__(self, name):
                if name in ('Dx', 'Dy', 'Dz', 'Diffx', 'Diffy', 'hMTF'):
                    if '_t' not in self.__dict__:
                        raise AssertionError('Surface fields read before initial snapshot')
                return super().__getattribute__(name)

            @OceanSurface.t.setter
            def t(self, value):
                updates.append(float(value))
                OceanSurface.t.fset(self, value)
                if not np.std(self.Dz) > 0:
                    raise AssertionError('Frozen snapshot must contain ocean relief')

        example = Path(__file__).resolve().parents[1]/'par'/'sim_example_rosel.cfg'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = ocs_io.ConfigFile(str(example))
            cfg.sim.seed = 12
            # This test inspects mocked surface calls in the current process.
            cfg.sim.memory_limit_gib = 0.
            cfg.sim.nworkers = 1
            cfg.sim.add_point_target = False
            cfg.sar.num_ch = 1
            cfg.RWV.burst_length[0] = .1
            cfg.ocean.Lx = cfg.ocean.Ly = 256.
            cfg.ocean.dx = cfg.ocean.dy = 8.
            cfg.ocean.frozen_ocean = True
            cfg.srg.factorize = False
            cfg.srg.scat_bragg_enable = False
            cfg.srg.scat_spec_enable = False
            cfg.srg.nesz = -5.  # Raw generation must stay noise-free even at high NESZ.
            cfg.save(str(root/'cfg.cfg'))
            for reuse in (False, True):
                with self.subTest(reuse_ocean=reuse):
                    updates.clear()
                    with patch('oceansar.radarsim.sar_raw.OceanSurface', CheckedSurface), \
                         contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        sar_raw(str(root/'cfg.cfg'), str(root/'raw.nc'),
                                str(root/'ocean.nc'), reuse, plot_save=True)
                    self.assertEqual(updates, [0.])
                    with Dataset(root/'raw.nc') as saved:
                        self.assertEqual(saved.raw_data_state, 'ideal')
                        self.assertEqual(saved.calibration_applied, 0)
                        self.assertGreater(float(saved['cal_factor'][...]), 0.)
                        np.testing.assert_array_equal(saved['raw_data_r'][:], 0.)
                        np.testing.assert_array_equal(saved['raw_data_i'][:], 0.)


if __name__ == '__main__':
    unittest.main()
