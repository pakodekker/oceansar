"""Point targets stay within the scene's intended mode image."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
from oceansar.radarsim.point_targets import point_target_indices
from oceansar.utils import geometry


class PointTargetPlacementTests(unittest.TestCase):
    def config(self, **mode):
        return NS(mode=NS(inc_angle=30., **mode), orbit=NS(Horb=693e3))

    def test_unbounded_mode_preserves_original_indices(self):
        x, y = np.linspace(-100, 100, 32), np.linspace(-200, 200, 64)
        self.assertEqual(point_target_indices(self.config(), x, y),
                         ((3, 16, 28), (6, 32, 57)))

    def test_image_length_caps_large_scene(self):
        axis = np.linspace(-1000, 1000, 201)
        x, y = point_target_indices(self.config(image_length=500.), axis, axis)
        self.assertEqual(x, (20, 100, 180))
        self.assertEqual(y, (80, 100, 120))

    def test_small_scene_is_not_expanded(self):
        axis = np.linspace(-100, 100, 21)
        self.assertEqual(point_target_indices(self.config(image_length=500.), axis, axis),
                         ((2, 10, 18), (2, 10, 18)))

    def test_selected_subswath_limits_range(self):
        cfg = self.config(definition='test', subswath=1)
        cfg.test = NS(inc_near=[20., 29.8], inc_far=[21., 30.3])
        axis = np.linspace(-20000, 20000, 401)
        xi, _ = point_target_indices(cfg, axis, axis)
        gr0 = geometry.inc_to_gr(np.deg2rad(30.), cfg.orbit.Horb)
        lower = geometry.inc_to_gr(np.deg2rad(29.8), cfg.orbit.Horb)-gr0
        upper = geometry.inc_to_gr(np.deg2rad(30.3), cfg.orbit.Horb)-gr0
        self.assertTrue(np.all(axis[list(xi)] >= lower))
        self.assertTrue(np.all(axis[list(xi)] <= upper))
        self.assertGreater(xi[0], 40)
        self.assertLess(xi[-1], 360)
