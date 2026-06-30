"""
=========================================
NRCS Models Module (:mod:`osiris.nrcs`)
=========================================
This module includes a number of backscattering models.

.. currentmodule:: osiris.nrcs

RCS Models

Available models
----------------
.. toctree::
   :maxdepth: 1

.. automodule:: osiris.nrcs.ka
   :members:

.. automodule:: osiris.nrcs.kodis
   :members:

.. automodule:: osiris.nrcs.romeiser97
   :members:
"""

from .ka import RCSKA
from .kodis import RCSKodis
from .romeiser97 import RCSRomeiser97

models = {'ka': RCSKA,
          'kodis': RCSKodis,
          'romeiser97': RCSRomeiser97}