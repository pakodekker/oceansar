"""Focused diagnostic positions for current-driven point targets."""
import unittest
import numpy as np
from oceansar.sar_l1.sar_processor import point_target_focused_displacement


class QuadraticHistory:
    def sr_spl(self, look, t):
        return np.asarray(800000. + 20.*t*t)

    def t2u_spl(self, look, t):
        return np.asarray(-.01*t)


class PointTargetMotionTests(unittest.TestCase):
    def test_stationary_targets_keep_their_positions(self):
        self.assertEqual(point_target_focused_displacement(
            QuadraticHistory(), .5, .5, .6, (0., 0.), 3., .2, 7000.), (0., 0.))

    def test_range_current_causes_doppler_azimuth_shift(self):
        az, rg = point_target_focused_displacement(
            QuadraticHistory(), .5, .5, .6, (2., 0.), 3., .2, 7000.)
        self.assertAlmostEqual(az, -7000.*1.2/40., delta=.01)
        self.assertAlmostEqual(rg, 3.6 - .2*1.2 - 1.2**2/80., delta=1e-5)

    def test_azimuth_current_changes_range_rate_and_range(self):
        elapsed, t, uy = 3., .2, 2.
        drift = -.01*uy*elapsed*t
        drift_rate = -.01*uy*(elapsed+t)
        matched = t+drift_rate/40.
        az, rg = point_target_focused_displacement(
            QuadraticHistory(), .5, .5, .6, (0., uy), elapsed, t, 7000.)
        self.assertAlmostEqual(az, 7000.*(t-matched), delta=.01)
        self.assertAlmostEqual(rg, drift+20.*(t*t-matched*matched), delta=1e-5)
