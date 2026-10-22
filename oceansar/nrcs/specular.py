"""Common specular field convention used by the simulator front ends."""

import numpy as np

from .ka import RCSKA
from .kodis import RCSKodis


def specular_field(model, azimuth, slant_range, incidence, height,
                   slope_x, slope_y, curvature_xx, curvature_yy, curvature_xy):
    """Evaluate one specular field shared by the HH and VV simulation channels.

    Parameters
    ----------
    model : RCSKodis or RCSKA
        Initialized specular model, including facet spacing and radar k0.
        KA supports its existing ``spa`` and ``fa`` modes.
    azimuth : float or ndarray
        Ground-projected azimuth angle in radians, broadcastable to the grid.
    slant_range : ndarray
        Propagation range in metres, with the caller's global origin and
        factorization reference already removed where applicable. Keep range
        precision through conversion to complex phase.
    incidence : float or ndarray
        Incidence angle in radians, normally a row spanning all range columns.
        Kodis uses the incidence supplied when its model was initialized.
    height, slope_x, slope_y : ndarray
        Grid heights in metres and dimensionless surface slopes.
    curvature_xx, curvature_yy, curvature_xy : ndarray
        Second height derivatives in inverse metres, on the same grid.

    Returns
    -------
    ndarray of complex
        One complex scattering field in simulator field units, with the
        existing Kodis/KA normalization and propagation-phase conventions.
        Add this SAME field to each enabled co-polarized scene, and compute
        its NRCS once before antenna/range weighting.

    Notes
    -----
    The simulators model the specular contribution as polarization independent.
    Kodis already provides a scalar field; for both KA modes evaluate VV once
    and share it with HH. RCSKA.field retains its general polarized interface
    for other uses. Bragg polarization dependence is unaffected.
    """
    if isinstance(model, RCSKodis):
        return np.sqrt(4.*np.pi) * model.field(
            azimuth, slant_range, slope_x, slope_y,
            curvature_xx, curvature_yy, curvature_xy)
    if isinstance(model, RCSKA):
        return (np.exp(-1j*(2.*model.k0*slant_range)) * (4.*np.pi)**1.5 *
                model.field(1, 1, 'v', 'v', incidence, incidence,
                            azimuth, azimuth + np.pi, height, slope_x, slope_y,
                            curvature_xx, curvature_yy, curvature_xy))
    raise TypeError('Unsupported specular model: %s' % type(model).__name__)
