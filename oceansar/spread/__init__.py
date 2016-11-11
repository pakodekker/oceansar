"""
=======================================================
Directional Spreading Functions (:mod:`osiris.spread`)
=======================================================

.. currentmodule:: osiris.spread

Directional Spreading Functions

All functions must be defined with the following parameters::

    model_name(k, theta, U_10, fetch)

Available models
----------------

.. toctree::
   :maxdepth: 1

.. automodule:: osiris.spread.banner
   :members:

.. automodule:: osiris.spread.cos2s
   :members:

.. automodule:: osiris.spread.elfouhaily
   :members:

.. automodule:: osiris.spread.hasselmann80
   :members:

.. automodule:: osiris.spread.mcdaniel
   :members:

.. automodule:: osiris.spread.romeiser97
   :members:

.. automodule:: osiris.spread.swop
   :members:
"""

from .banner import *
from .cos2s import *
from .elfouhaily import *
from .hasselmann80 import *
from .mcdaniel import *
from .romeiser97 import *
from .swop import *


models = {'banner': banner,
          'cos2s': cos2s,
          'elfouhaily': elfouhaily,
          'hasselmann80': hasselmann80,
          'mcdaniel': mcdaniel,
          'romeiser97': romeiser97,
          'swop': swop,
          'none': None}
