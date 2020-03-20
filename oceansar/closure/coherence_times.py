
import numpy as np
from scipy import special
import oceansar.constants as const


def grid_coherence(U, res, f0, model='Pierson-Moskowitz'):
    """ Calculates the coherence time of a resolution cell

        :param U: wind velocity at reference height
        :param res: grid size [m]
        :param f0: carrier frequency [Hz]
        :param model: Model used, defaults to Pierson-Moskowitz (currently
                      the only one implemented)
    """

    tau_c = 3.29 * const.c / f0 / U / np.sqrt(special.erf(2.688 * res / U**2))
    return tau_c
