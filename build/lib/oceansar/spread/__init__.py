"""
=======================================================
Directional Spreading Functions (:mod:`oceansar.spread`)
=======================================================

.. currentmodule:: oceansar.spread

Directional Spreading Functions

All functions must be defined with the following parameters::

    model_name(k, theta, U_10, fetch)

Available models
----------------

.. toctree::
   :maxdepth: 1

.. automodule:: oceansar.spread.banner
   :members:

.. automodule:: oceansar.spread.cos2s
   :members:

.. automodule:: oceansar.spread.elfouhaily
   :members:

.. automodule:: oceansar.spread.hasselmann80
   :members:

.. automodule:: oceansar.spread.mcdaniel
   :members:

.. automodule:: oceansar.spread.romeiser97
   :members:

.. automodule:: oceansar.spread.swop
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
