"""Ocean Level-2 processing algorithms, shared unchanged between projects.

The public functions accept arrays and explicit physical parameters; readers,
configuration handling and workflow orchestration belong to the host project.
"""

from .cross_spectrum import outlier_removal, quick_xspec, welch_xspec
from .dc_estimator import estimate_dc
from .multilook import multilook_report, multilook_window

__all__ = ['estimate_dc', 'multilook_window', 'multilook_report',
           'welch_xspec', 'quick_xspec', 'outlier_removal']
