
import numpy as np

from oceansar import constants as const

class RCSKA():

    """ Kirchoff Approximation

        Available solution modes: SPA / FA

        - SPA: Stationary-Phase Method

        - FA: Facet Approach E.M. calculation model
              This models calculates the E.M. of a surface
              solving the Stratton-Chu integral on each facet.
              This way, an explicit solution for the integral
              can be implemented and hence leading to reasonable
              computation times.

        References:
            "Simulation of L-Band Bistatic Returns From the Ocean Surface:
             A Facet Approach With Application to Ocean GNSS Reflectometry"
             M. Paola Clarizia et al.

             "Microwave Remote Sensing. Active and Passive (Vol II)" Sect. 12-4
             T. Ulaby, R. Moore, A. Fung

        :param mode: Computation mode (spa, fa)
        :param k0: Radar wave number
        :param x: Surface 'X' vector
        :param y: Surface 'Y' vector
        :param dx: Surface 'X' facet resolution
        :param dy: Surface 'Y' facet resolution

    """

    def __init__(self, mode, k0, x, y, dx, dy):

        # Save values
        self.mode = mode
        self.k0 = k0
        self.dx = dx
        self.dy = dy
        self.shape = [y.shape[0], x.shape[0], 3]

        # r (local position) vector
        self.r = np.empty(self.shape)
        self.r[:, :, 0], self.r[:, :, 1] = np.meshgrid(x, y)


#    def field_mono(self, R_i, pol_i, pol_s, theta_i, phi_i,
#                   Dz, Diffx, Diffy, Diffxx, Diffyy, Diffxy):
#        """
#        Calculates E.M. field in monostatic case
#        :param R_i: Distance from transmitter to scene center
#        :param pol_i: Incident polarization (v, h)
#        :param pol_s: Scattered polarization (v, h)
#        :param theta_i: Incidence elevation angle
#        :param phi_i: Incidence azimuth angle
#        :param Dz: Surface height field
#        :param Diffx: Space first derivatives (X slopes)
#        :param Diffy: Space first derivatives (Y slopes)
#        :param Diffxx: Space second derivatives (XX)
#        :param Diffyy: Space second derivatives (YY)
#        :param Diffxy: Space second derivatives (XY)
#        """
#        sin_theta_i = np.sin(theta_i)
#        cos_theta_i = np.cos(theta_i)
#        tan_theta_i = np.cos(theta_i)
#        sin_phi_i = np.sin(phi_i)
#        cos_phi_i = np.cos(phi_i)
#
#        if self.mode == 'spa':
#            J = Diffxx*Diffyy - Diffxy**2
#            J = np.where(J == 0., np.nan, J)
#            J_abs = np.abs(J)
#            delta_x = (1./J_abs)*(Diffxy*(Diffy - tan_theta_i * sin_phi_i) -
#                                  Diffyy*(Diffx - tan_theta_i * cos_phi_i))
#            delta_y = (1./J_abs)*(Diffxy*(Diffx - tan_theta_i * sin_phi_i) -
#                                  Diffxx*(Diffy - tan_theta_i * sin_phi_i))
#            delta_z = delta_x * Diffx + delta_y * Diffy
#            epsilon = np.where(J > 0., np.sign(Diffxx), 1j)
#
#            E = np.zeros(delta_x.shape, dtype=np.complex)
#            sps = np.where(((-0.5 * self.dx < delta_x) &
#                            (delta_x < 0.5 * self.dx)) &
#                           ((-0.5 * self.dy < delta_y) &
#                            (delta_y < 0.5 * self.dy)))
#
#            if sps[0].size > 0:
#                p = np.sum(a_s *
#                           np.cross(n_s,
#                                    n__x__Es -
#                                    np.cross(n_s, etha__p__n__x__Hs)),
#                           axis=-1)
#
#                E[sps] = (epsilon[sps] * p[sps] * K *
#                          np.exp(-1j * (q[..., 0][sps] * delta_x[sps] +
#                                        q[..., 1][sps] * delta_y[sps] +
#                                        q[..., 2][sps] * delta_z[sps])) *
#                          2.*np.pi/q[..., 2][sps] * n_norm[sps] /
#                          np.sqrt(J_abs[sps]))


    def field(self, R_i, R_s,
                    pol_i, pol_s,
                    theta_i, theta_s, phi_i, phi_s,
                    Dz, Diffx, Diffy, Diffxx, Diffyy, Diffxy,
                    monostatic = True):

        """ Calculates E.M. field

            :param R_i: Distance from transmitter to scene center
            :param R_s: Distance from scene center to receiver
            :param pol_i: Incident polarization (v, h)
            :param pol_s: Scattered polarization (v, h)
            :param theta_i: Incidence elevation angle
            :param theta_s: Scattered elevation angle
            :param phi_i: Incidence azimuth angle
            :param phi_s: Scattered azimuth angle
            :param Dz: Surface height field
            :param Diffx: Space first derivatives (X slopes)
            :param Diffy: Space first derivatives (Y slopes)
            :param Diffxx: Space second derivatives (XX)
            :param Diffyy: Space second derivatives (YY)
            :param Diffxy: Space second derivatives (XY)
            :param monostatic: Tue for mono geometry

        """

        ### CACHE ###
        sin_theta_i = np.sin(theta_i).reshape((1, theta_i.size))
        # sin_theta_i = np.repeat(sin_theta_i, self.shape[0], axis=0)
        cos_theta_i = np.cos(theta_i).reshape((1, theta_i.size))
        # cos_theta_i = np.repeat(cos_theta_i, self.shape[0], axis=0)
        tan_theta_i = np.tan(theta_i) # This was sin, I guess this was a bug!
        sin_phi_i = np.sin(phi_i)
        cos_phi_i = np.cos(phi_i)
        h_i = np.empty(self.shape)
        h_i[:, :, 0] = -sin_phi_i
        h_i[:, :, 1] = cos_phi_i
        h_i[:, :, 2] = 0.
        v_i = np.empty(self.shape)
        v_i[:, :, 0] = -cos_theta_i*cos_phi_i
        v_i[:, :, 1] = -cos_theta_i*sin_phi_i
        v_i[:, :, 2] = -sin_theta_i
        if monostatic:
            sin_theta_s = sin_theta_i
            cos_theta_s = cos_theta_i
            sin_phi_s = sin_phi_i
            cos_phi_s = cos_phi_i
            h_s = h_i
            v_s = v_i
        else:
            sin_theta_s = np.sin(theta_s).reshape((1, theta_i.size))
            # sin_theta_s = np.repeat(sin_theta_s, self.shape[0], axis=0)
            cos_theta_s = np.cos(theta_s).reshape((1, theta_i.size))
            # cos_theta_s = np.repeat(cos_theta_s, self.shape[0], axis=0)
            sin_phi_s = np.sin(phi_s)
            cos_phi_s = np.cos(phi_s)
            h_s = np.empty(self.shape)
            h_s[:, :, 0] = -sin_phi_s
            h_s[:, :, 1] = cos_phi_s
            h_s[:, :, 2] = 0.
            v_s = np.empty(self.shape)
            v_s[:, :, 0] = cos_theta_s*cos_phi_s
            v_s[:, :, 1] = cos_theta_s*sin_phi_s
            v_s[:, :, 2] = -sin_theta_s
        ### VECTORS ###
        # Position (r) - Update heights
        self.r[:, :, 2] = Dz

        # Polarization vectors (H, V)

        a_i = h_i if pol_i == 'h' else v_i
        a_s = h_s if pol_s == 'h' else v_s

        # Surface normal (n)
        n = np.empty(self.shape)
        n_norm = np.sqrt(Diffx**2. + Diffy**2. + 1.)
        n[:, :, 0] = -Diffx/n_norm
        n[:, :, 1] = -Diffy/n_norm
        n[:, :, 2] = 1./n_norm

        # Incidence direction (n_i)
        n_i = np.empty(self.shape)
        n_i[:, :, 0] = sin_theta_i * cos_phi_i
        n_i[:, :, 1] = sin_theta_i * sin_phi_i
        n_i[:, :, 2] = -cos_theta_i

        # Scattering direction (n_s)
        n_s = np.empty(self.shape)
        # FIXME
        # Changed the sign of the first two terms, since I think it was wrong!
        n_s[:, :, 0] = - sin_theta_s * cos_phi_s
        n_s[:, :, 1] = - sin_theta_s * sin_phi_s
        n_s[:, :, 2] = cos_theta_s

        # Scattering (q)
        q = self.k0*(n_s - n_i)

        # Local frame of reference (t, d)
        t = np.cross(n_i, n)
        t /= np.sqrt(np.sum(t**2, axis=2)).reshape((self.shape[0], self.shape[1], 1))
        d = np.cross(n_i, t)


        ### E.M. Field ###
        # Fresnel reflection coefficients
        cos_theta_l = -np.sum(n*n_i, axis=-1)
        sqrt_e_sin = np.sqrt(const.epsilon_sw + cos_theta_l**2. - 1.)
        r_v = ((const.epsilon_sw*cos_theta_l - sqrt_e_sin)/
               (const.epsilon_sw*cos_theta_l + sqrt_e_sin))
        r_h = ((cos_theta_l - sqrt_e_sin)/
               (cos_theta_l + sqrt_e_sin))


        # Scattered E/H fields and projected 'p' vector
        a__P__t = np.sum(a_i*t, axis=-1)
        a__P__d = np.sum(a_i*d, axis=-1)
        n__P__n_i = -cos_theta_l
        n__x__t = np.cross(n, t)

        n__x__Es = (((1. + r_h)*a__P__t).reshape((self.shape[0], self.shape[1], 1)) * n__x__t -
                    ((1. - r_v)*n__P__n_i*a__P__d).reshape((self.shape[0], self.shape[1], 1)) * t)
        etha__p__n__x__Hs = -(((1. - r_h)*n__P__n_i*a__P__t).reshape((self.shape[0], self.shape[1], 1)) * t +
                              ((1. + r_v)*a__P__d).reshape((self.shape[0], self.shape[1], 1)) * n__x__t)

        p = np.sum(a_s*np.cross(n_s, n__x__Es - np.cross(n_s, etha__p__n__x__Hs)), axis=-1)

        #K
        K = -1j*self.k0*np.exp(-1j*self.k0*(R_i + R_s))/((4.*np.pi)**2.*R_i*R_s)


        # Stationary-Phase Approximation
        if self.mode == 'spa':
            J = Diffxx*Diffyy - Diffxy**2
            J = np.where(J == 0., np.nan, J)
            J_abs = np.abs(J)
            delta_x = (1./J_abs)*(Diffxy*(Diffy - tan_theta_i*sin_phi_i) - Diffyy*(Diffx - tan_theta_i*cos_phi_i))
            delta_y = (1./J_abs)*(Diffxy*(Diffx - tan_theta_i*cos_phi_i) - Diffxx*(Diffy - tan_theta_i*sin_phi_i))
            delta_z = delta_x * Diffx + delta_y * Diffy
            epsilon = np.where(J > 0., np.sign(Diffxx), 1j)

#            E = np.where(((0. < delta_x) & (delta_x < self.dx)) & ((0. < delta_y) & (delta_y < self.dy)),
#                         epsilon*p*K*np.exp(-1j*np.sum(q*self.r, axis=-1))*2.*np.pi/q[..., 2]*n_norm/np.sqrt(J_abs),
#                         0.)
#            E = np.where(((-0.5 * self.dx < delta_x) &
#                          (delta_x < 0.5 * self.dx)) &
#                         ((-0.5 * self.dy < delta_y) &
#                          (delta_y < 0.5 * self.dy)),
#                         (epsilon*p*K*np.exp(-1j*np.sum(q*self.r, axis=-1)) *
#                          2.*np.pi/q[..., 2]*n_norm/np.sqrt(J_abs)),
#                         0.)
            E = np.zeros(delta_x.shape, dtype=np.complex)
            sps = np.where(((-0.5 * self.dx < delta_x) &
                            (delta_x < 0.5 * self.dx)) &
                           ((-0.5 * self.dy < delta_y) &
                            (delta_y < 0.5 * self.dy)))

            if sps[0].size > 0:
                E[sps] = (epsilon[sps] * p[sps] * K *
                          np.exp(1j * (q[..., 0][sps] * delta_x[sps] +
                                       q[..., 1][sps] * delta_y[sps] +
                                       q[..., 2][sps] * delta_z[sps])) *
                          2.*np.pi/q[..., 2][sps] * n_norm[sps] /
                          np.sqrt(J_abs[sps]))


        # Facet Approach
        else:
            # Integral
            I = (np.exp(-1j*np.sum(q*self.r, axis=-1))*n_norm* self.dx * self.dy *
                 np.sinc((q[..., 0] + q[..., 2]*Diffx)*self.dx/2.)*
                 np.sinc((q[..., 1] + q[..., 2]*Diffy)*self.dy/2.))

            # Scattered field
            E = K*I*p


        return E


    def rcs(self, pol_i, pol_s,
            theta_i, theta_s, phi_i, phi_s,
            Dz, Diffx, Diffy, Diffxx, Diffyy, Diffxy):

        """ Calculates RCS

            :param pol_i: Incident polarization (v, h)
            :param pol_s: Scattered polarization (v, h)
            :param theta_i: Incidence elevation angle
            :param theta_s: Scattered elevation angle
            :param phi_i: Incidence azimuth angle
            :param phi_s: Scattered azimuth angle
            :param Dz: Surface height field
            :param Diffx: Space first derivatives (X slopes)
            :param Diffy: Space first derivatives (Y slopes)
            :param Diffxx: Space second derivatives (XX)
            :param Diffyy: Space second derivatives (YY)
            :param Diffxy: Space second derivatives (XY)

        """

        return (4.*np.pi)**2.*np.abs(self.field(1, 1, pol_i, pol_s,
                                                theta_i, theta_s, phi_i, phi_s,
                                                Dz, Diffx, Diffy, Diffxx, Diffyy, Diffxy))**2.
