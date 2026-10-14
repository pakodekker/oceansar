import unittest
from types import SimpleNamespace

import numpy as np

from oceansar.sar_l1.sar_processor import (
    scansar_azimuth_focus,
    scansar_deramp_phase,
    geohistory_doppler_sine,
)


class ScanSARDerampTests(unittest.TestCase):
    v_ground = 6829.443994
    v_eff = 7200.086177
    wavelength = .238403545
    ranges = np.array([782975.13646654, 788031.05967981])

    def phase(self, size, start_time, prf):
        return scansar_deramp_phase(
            size, self.ranges, start_time * self.v_ground, prf,
            self.v_ground, self.v_eff, self.wavelength,
        )

    def test_antenna_angle_uses_orbital_projection_in_fft_order(self):
        orbital_velocity = 7590.981
        frequencies = np.fft.fftfreq(16, 1/1378.)

        def doppler_to_geometry(look_angle, sorted_frequencies, f0):
            self.assertTrue(np.all(np.diff(sorted_frequencies) < 0))
            sine = sorted_frequencies * self.wavelength / (2 * orbital_velocity)
            return None, sine[None, :], None, None, None

        geometry = SimpleNamespace(Doppler2tuv=doppler_to_geometry)
        sine = geohistory_doppler_sine(geometry, .5, frequencies, 1.2575e9)
        np.testing.assert_allclose(
            sine, frequencies * self.wavelength / (2 * orbital_velocity),
        )
        self.assertFalse(np.allclose(
            sine, frequencies * self.wavelength / (2 * self.v_eff),
        ))

    def test_integrated_range_phase_is_even_and_zero_at_scene_center(self):
        times = np.arange(-2, 3) / 4
        phase = self.phase(5, -.5, 4)
        instantaneous_range = np.hypot(
            self.ranges[None, :], self.v_eff * times[:, None],
        )
        expected = 4 * np.pi / self.wavelength * (
            instantaneous_range - self.ranges[None, :]
        )
        np.testing.assert_allclose(phase, expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(phase, phase[::-1], rtol=0, atol=0)
        np.testing.assert_array_equal(phase[2], 0)

    def test_quadratic_limit_remains_nonzero_near_scene_center(self):
        # Direct subtraction of these nearly equal ranges rounds to zero.
        times = np.array([-1e-6, 0, 1e-6])
        phase = self.phase(3, -1e-6, 1e6)
        quadratic = (2 * np.pi * self.v_eff**2 * times[:, None]**2
                     / (self.wavelength * self.ranges[None, :]))
        self.assertTrue(np.all(phase[[0, 2]] > 0))
        np.testing.assert_allclose(phase, quadratic, rtol=1e-12, atol=0)

    def test_deramping_removes_geometric_chirp_without_residual_doppler(self):
        size = 4097
        half_span = 10000 / (2 * self.v_ground)
        prf = (size - 1) / (2 * half_span)
        times = -half_span + np.arange(size) / prf
        geometric_phase = 4 * np.pi / self.wavelength * (
            np.hypot(self.ranges[None, :], self.v_eff * times[:, None])
            - self.ranges[None, :]
        )
        spectrum = np.fft.fft(np.exp(1j * geometric_phase)[None, :, :], axis=1)
        focused = scansar_azimuth_focus(
            spectrum, np.zeros(size), self.phase(size, -half_span, prf),
            reramp=False,
        )[0]
        np.testing.assert_allclose(focused, 1, rtol=0, atol=1e-8)
        residual_hz = np.angle(focused[1:] * focused[:-1].conj()) * prf / (2 * np.pi)
        np.testing.assert_allclose(residual_hz, 0, rtol=0, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
