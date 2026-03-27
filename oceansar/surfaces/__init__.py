"""
========================================
Ocean surfaces (:mod:`osiris.surfaces`)
========================================

.. currentmodule:: osiris.surfaces

This module contains ocean surface class and related utilities

Ocean surfaces
--------------
.. toctree::
   :maxdepth: 2

.. automodule:: osiris.surfaces.ocean
   :members:

.. automodule:: osiris.surfaces.balancer
   :members:
"""

from .ocean import OceanSurface
try:
    from .balancer import OceanSurfaceBalancer
except:
    print("OceanSurfaceBalancer import failed, probably no mpi4py")
