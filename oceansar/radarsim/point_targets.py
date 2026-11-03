"""Shared point-target placement on the simulated surface grid."""
import numpy as np

from oceansar.ocs_io.sar_config import selected
from oceansar.utils import geometry


def point_target_indices(cfg, x, y):
    """Return three range and azimuth indices inside the intended image.

    Intersect the scene with the centred mode image_length and the selected
    subswath's incidence limits. Retain the usual 10%, centre, 90% placement
    on the available grid; modes without limits use the full scene.
    """
    x, y = np.asarray(x), np.asarray(y)
    xlo, xhi = x[0], x[-1]
    ylo, yhi = y[0], y[-1]
    length = getattr(cfg.mode, 'image_length', None)
    if length is not None:
        ylo, yhi = max(ylo, -length/2), min(yhi, length/2)
    definition = getattr(cfg, (getattr(cfg.mode, 'definition', None) or ''), cfg.mode)
    if hasattr(definition, 'inc_near') and hasattr(definition, 'inc_far'):
        index = int(getattr(cfg.mode, 'subswath', 0))
        near = selected(definition.inc_near, index, 'inc_near')
        far = selected(definition.inc_far, index, 'inc_far')
        gr0 = geometry.inc_to_gr(np.deg2rad(cfg.mode.inc_angle), cfg.orbit.Horb)
        xlo = max(xlo, geometry.inc_to_gr(np.deg2rad(near), cfg.orbit.Horb)-gr0)
        xhi = min(xhi, geometry.inc_to_gr(np.deg2rad(far), cfg.orbit.Horb)-gr0)

    def indices(axis, lower, upper):
        inside = np.flatnonzero((axis >= lower) & (axis <= upper))
        if not inside.size:
            raise ValueError('Intended point-target image contains no surface grid samples')
        first, last = int(inside[0]), int(inside[-1])
        edge = int(round(.1*(last-first)))
        return first+edge, (first+last+1)//2, last-edge

    return indices(x, xlo, xhi), indices(y, ylo, yhi)
