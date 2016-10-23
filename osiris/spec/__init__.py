"""
=====================================
Spectrum Models (:mod:`osiris.spec`)
=====================================

.. currentmodule:: osiris.spec

This module contains omni-directional wavespectrum models

.. note::
    All ocean spectrum functions must be defined with the following parameters::

        model_name(k, U_10, fetch)

Ocean spectrum models
---------------------
.. toctree::
   :maxdepth: 1

.. automodule:: osiris.spec.elfouhaily
   :members:

.. automodule:: osiris.spec.jonswap
   :members:

.. automodule:: osiris.spec.neumann_
   :members:

.. automodule:: osiris.spec.romeiser97
   :members:
"""

from .elfouhaily import *
from .jonswap import *
#from neumann import *
from .romeiser97 import *

models = {'elfouhaily': elfouhaily,
          'jonswap': jonswap,
          #'neumann': neumann,
          'romeiser97': romeiser97}