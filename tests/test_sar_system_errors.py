import ast
import contextlib
import json
import os
from types import SimpleNamespace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

from oceansar import ocs_io
from oceansar.radarsim.sar_system_errors import sar_system_errors, validate_raw_paths


class SARSystemErrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = self.root/'sim.cfg'
        self.cfg.write_text(f'[sim]\npath={self.root}\nseed=43\n'
                            'raw_file=ideal.nc\nsys_raw_file=system.nc\nproc_file=slc.nc\n'
                            'raw_run=False\nproc_run=False\ninsar_run=False\n'
                            'ati_run=False\nL2_wavespectrum_run=False\n'
                            '[mode]\npol=DP\n[srg]\nnesz=-20\n')
        self.ideal = self.root/'ideal.nc'
        self.system = self.root/'system.nc'
        shape = (2, 2, 101, 257)
        rng = np.random.default_rng(7)
        self.raw = rng.normal(size=shape) + 1j*rng.normal(size=shape)
        f = ocs_io.RawFile(str(self.ideal), 'w', shape)
        f.set('raw_data*', self.raw)
        f.set('prf', 1378.)
        f.set('az0', -123.)
        f.set('b_ati', [0., 2.2])
        f.set('sr_pt', np.arange(shape[2], dtype=float))
        f.set('NRCS_avg', np.ones((shape[0], shape[2])))
        f.__file__.raw_data_state = 'ideal'
        f.__file__.calibration_applied = 0
        f.__file__.custom_metadata = 'preserve this'
        f.__file__.createVariable('cal_factor', 'f8').assignValue(2.5)
        f.close()
        self.original_bytes = self.ideal.read_bytes()

    def read(self, path):
        with Dataset(path) as f:
            return f['raw_data_r'][:] + 1j*f['raw_data_i'][:]

    def test_calibration_noise_metadata_and_input_preservation(self):
        sar_system_errors(self.cfg, self.ideal, self.system, block_azimuth=17)
        error = self.read(self.system) - 2.5*self.raw
        self.assertLess(abs(error.mean()), .002)
        self.assertAlmostEqual(float(np.mean(abs(error)**2)), .01, delta=.0002)
        self.assertAlmostEqual(float(np.var(error.real)), .005, delta=.00015)
        self.assertAlmostEqual(float(np.var(error.imag)), .005, delta=.00015)
        self.assertLess(abs(np.mean(error[0]*error[1].conj())), .0002)
        self.assertEqual(self.ideal.read_bytes(), self.original_bytes)
        with Dataset(self.ideal) as source, Dataset(self.system) as target:
            self.assertEqual(target.raw_data_state, 'system_errors')
            self.assertEqual(target.calibration_applied, 1)
            self.assertEqual(target.nesz_db, -20.)
            self.assertEqual(target.noise_seed, '43')
            self.assertEqual(target.custom_metadata, 'preserve this')
            for name, variable in source.variables.items():
                self.assertEqual(target[name].dtype, variable.dtype)
                self.assertEqual(target[name].dimensions, variable.dimensions)
                if name not in ('raw_data_r', 'raw_data_i'):
                    np.testing.assert_array_equal(target[name][:], variable[:])

    def test_seed_and_block_size_independence(self):
        sar_system_errors(self.cfg, self.ideal, self.system, block_azimuth=13)
        expected = self.read(self.system)
        for size in (1, 101, 128):
            output = self.root/f'system_{size}.nc'
            sar_system_errors(self.cfg, self.ideal, output, block_azimuth=size)
            np.testing.assert_array_equal(self.read(output), expected)

    def test_calibration_with_noise_disabled(self):
        self.cfg.write_text(self.cfg.read_text().replace('nesz=-20', 'nesz=-inf'))
        sar_system_errors(self.cfg, self.ideal, self.system)
        np.testing.assert_array_equal(self.read(self.system), 2.5*self.raw)

    def test_rejects_nonideal_inputs_and_same_output(self):
        with self.assertRaises(ValueError):
            sar_system_errors(self.cfg, self.ideal, self.ideal)
        with Dataset(self.ideal, 'r+') as f:
            f.raw_data_state = 'system_errors'
        with self.assertRaisesRegex(ValueError, 'Regenerate ideal'):
            sar_system_errors(self.cfg, self.ideal, self.system)
        self.assertFalse(self.system.exists())

    def run_entrypoint(self, entrypoint, focus, enabled=True):
        if entrypoint == 'simulator':
            from oceansar.radarsim.oceansar_sarsim import sarsim
            text = self.cfg.read_text().replace('proc_run=False', 'proc_run=True')
            if not enabled:
                text = text.replace('proc_run=True', 'proc_run=False')
            self.cfg.write_text(text)
            with patch('oceansar.radarsim.oceansar_sarsim.sar_focus', side_effect=focus):
                sarsim(str(self.cfg))
        else:
            # Execute the notebook's actual orchestration block without running
            # its interactive setup, raw generator or downstream analysis cells.
            notebook = Path(__file__).resolve().parents[1]/'Notebooks'/'sarsim.ipynb'
            cells = json.loads(notebook.read_text())['cells']
            source = next(''.join(c['source']) for c in cells
                          if c['cell_type'] == 'code' and 'if do_focus:' in ''.join(c['source']))
            block = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                         and node.test.id == 'do_focus')
            code = compile(ast.Module(body=[block], type_ignores=[]), str(notebook), 'exec')
            exec(code, {'do_focus': enabled, 'cfg': SimpleNamespace(cfg_file_name=str(self.cfg)),
                        'raw_file': str(self.ideal), 'sys_raw_file': str(self.system),
                        'slc_file': str(self.root/'slc.nc'), 'os': os,
                        'sar_system_errors': sar_system_errors, 'sar_focus': focus,
                        'validate_raw_paths': validate_raw_paths})

    def test_entrypoints_focus_system_file_and_delete_it_after_success(self):
        calls = []
        def focus(config, raw_path, output):
            calls.append('focus')
            self.assertEqual(Path(raw_path), self.system)
            self.assertTrue(self.system.exists())
            with Dataset(raw_path) as f:
                self.assertEqual(f.raw_data_state, 'system_errors')
            Path(output).write_text('focused')
        for entrypoint in ('simulator', 'notebook'):
            with self.subTest(entrypoint=entrypoint), contextlib.redirect_stdout(io.StringIO()):
                self.run_entrypoint(entrypoint, focus)
                self.assertTrue((self.root/'slc.nc').exists())
                self.assertFalse(self.system.exists())
                self.assertEqual(self.ideal.read_bytes(), self.original_bytes)
        self.assertEqual(calls, ['focus', 'focus'])

    def test_entrypoints_retain_system_file_when_focusing_fails(self):
        def fail(*args):
            raise RuntimeError('focus failed')
        for entrypoint in ('simulator', 'notebook'):
            with self.subTest(entrypoint=entrypoint), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(RuntimeError, 'focus failed'):
                self.run_entrypoint(entrypoint, fail)
            self.assertTrue(self.system.exists())
            self.assertEqual(self.ideal.read_bytes(), self.original_bytes)

    def test_entrypoints_skip_system_errors_and_focusing_when_disabled(self):
        def fail(*args):
            self.fail('Focusing must not run when disabled')
        for entrypoint in ('simulator', 'notebook'):
            with self.subTest(entrypoint=entrypoint), contextlib.redirect_stdout(io.StringIO()):
                self.run_entrypoint(entrypoint, fail, enabled=False)
                self.assertFalse(self.system.exists())
                self.assertFalse((self.root/'slc.nc').exists())

    def test_entrypoints_reject_output_overwriting_ideal_raw(self):
        def fail(*args):
            self.fail('Focusing must not run with conflicting filenames')
        self.cfg.write_text(self.cfg.read_text().replace('proc_file=slc.nc', 'proc_file=ideal.nc'))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.run_entrypoint('simulator', fail)
        self.assertFalse(self.system.exists())
        self.assertEqual(self.ideal.read_bytes(), self.original_bytes)


if __name__ == '__main__':
    unittest.main()
