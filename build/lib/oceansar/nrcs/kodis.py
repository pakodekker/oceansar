import numpy as np

from oceansar import constants as const

class RCSKodis():
    """ Specular model (R.D. Kodis '66)

        Physical optics model as described in R.D. Kodis (1966) paper
        'A Note on the Theory of Scattering from an Irregular Surface'.
        E.M. solved using Stationary Phase Method.

        .. note::
            G. Valenzuela suggested that reflection coefficient (R)
            may be replaced by effective refl. coef.!

        .. note::
            OASIS uses only range dependent incidence angle, so
            it is given on class init.

        :param inc: Incidence angle matrix
        :param k0: Radar wave number
        :param dx: Range resolution
        :param dy: Azimuth resolution

    """

    def __init__(self, inc, k0, dx, dy):
        self.dx = dx
        self.dy = dy
        self.k0 = k0

        self.sin_inc = np.sin(inc)
        self.cos_inc = np.cos(inc)
        self.tan_inc = np.tan(inc)

        self.R = (const.epsilon_sw - np.sqrt(const.epsilon_sw))/(const.epsilon_sw + np.sqrt(const.epsilon_sw))

    def field(self, az_angle, sr, diffx, diffy, diffxx, diffyy, diffxy):

        # Avoid repeating calculations
        cos_az = np.cos(az_angle)
        sin_az = np.sin(az_angle)

        J = diffxx*diffyy - diffxy**2
        J = np.where(J == 0., np.nan, J)
        J_abs = np.abs(J)
        delta_x = (1./J_abs)*(diffxy*(diffy - self.tan_inc*sin_az) - diffyy*(diffx - self.tan_inc*cos_az))
        delta_y = (1./J_abs)*(diffxy*(diffx - self.tan_inc*cos_az) - diffxx*(diffy - self.tan_inc*sin_az))

        epsilon = np.where(J > 0., np.sign(diffxx), 1j)

        # New slant range due to deltas
        hdx = self.dx/2
        hdy = self.dy/2
        E = np.zeros(delta_x.shape, dtype=np.complex)
        sps = np.where(((-hdx < delta_x) & (delta_x < hdx)) &
                       ((-hdy < delta_y) & (delta_y < hdy)))
        if sps[0].size > 0:
            delta_z = delta_x[sps] * diffx[sps] + delta_y[sps] * diffy[sps]
            sr_p2 = (sr[sps] +
                     (self.sin_inc[0,sps[1]] * cos_az[sps] * delta_x[sps] +
                      self.sin_inc[0,sps[1]] * sin_az[sps] * delta_y[sps] -
                      self.cos_inc[0,sps[1]] * delta_z))
            E[sps] = ((0.5*self.R*epsilon[sps]) *
                      ((diffx[sps]**2. + diffy[sps]**2. + 1.)) *
                      np.exp(-1j*2.*self.k0*sr_p2) /
                      np.sqrt(J_abs[sps]))

#        field = np.where(((-hdx < delta_x) & (delta_x < hdx)) & ((-hdy < delta_y) & (delta_y < hdy)),
#                         (0.5*self.R*epsilon)*((diffx**2. + diffy**2. + 1.)) * np.exp(-1j*2.*self.k0*np.sqrt(sr_p2)) / np.sqrt(J_abs),
#                         0.)

        return E

    def candidates(self, az_angle, diffx, diffy, diffxx, diffyy, diffxy):

        # Avoid repeating calculations
        cos_az = np.cos(az_angle)
        sin_az = np.sin(az_angle)

        J = diffxx*diffyy - diffxy**2.
        J = np.where(J == 0., np.nan, np.abs(J))
        delta_x = (1./J)*(diffxy*(diffy - self.sin_inc*sin_az) - diffyy*(diffx - self.sin_inc*cos_az))
        delta_y = (1./J)*(diffxy*(diffx - self.sin_inc*cos_az) - diffxx*(diffy - self.sin_inc*sin_az))

        candidates = np.where(((0. < delta_x) & (delta_x < self.dx)) & ((0. < delta_y) & (delta_y < self.dy)),
                              1., 0.)

        return candidates