"""Compatibility names for the shared implementations in :mod:`drama.utils`.

OceanSAR-specific random streams and geometry live in their own modules. Keep
these small argument adapters so existing callers use DRAMA's algorithms too.
"""

from drama import utils as _drama
from drama.utils.misc import nearest_power2
from drama.utils.filtering import smooth1d

factorize = _drama.factorize
balance_elements = _drama.balance_elements
optimize_fftsize = _drama.optimize_fftsize
get_parFile = _drama.get_par_file
PrInfo = _drama.PrInfo
linresample = _drama.linresample
db2lin = _drama.db2lin


def db(a, linear=False):
    """Convert power to dB, or amplitude when the historical linear flag is set."""
    return (2 if linear else 1) * _drama.db(a)


def smooth(data, window_len_=11, window='flat', axis=None, force_fft=False):
    """Accept OceanSAR's historical window-length spelling and use DRAMA."""
    return _drama.smooth(data, int(window_len_), window, axis, force_fft)
