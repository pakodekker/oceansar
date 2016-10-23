"""
===========================================
Closure Module (:mod:`osiris.closure`)
===========================================
This module includes routines to model what happens at sub-grid scales.

.. currentmodule:: osiris.closure

Constants
=========

.. toctree::
   :maxdepth: 2

.. automodule:: osiris.closure.coherence_times
   :members:

.. automodule:: osiris.closure.random_scatterers
   :members:
"""

from .coherence_times import grid_coherence
from .random_scatterers import randomscat_ts