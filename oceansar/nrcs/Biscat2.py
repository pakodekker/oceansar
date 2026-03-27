import numpy as np
from scipy import interpolate

from oceansar import spec
from oceansar import spread

from oceansar import constants as const
import numexpr as ne

class Bistaticscat():
    """ Bistatic SPM1 + SPM2 
        This function returns RCS using first and second order small perturbation methods
       

        :param k0: Radar wave number
        :param theta_i: Incidence elevation angle
        :param theta_s: Scattered elevation angle
        :param phi_i: Incidence azimuth angle
        :param phi_s: Scattered azimuth angle
        :param pol_i: Incident polarization (v, h)
        :param pol_s: Scattered polarization (v, h)
        :param dx: Surface facets X size
        :param dy: Surface facets Y size
        :param wind_U: Wind speed (U_10)
        :param wind_dir: Wind blowing direction (rad)
        :param wind_fetch: Wind fetch
        :param sw_spec_func: Omnidirectional spectrum function
        :param sw_spread_func: Spreading function
        :param d: Filter threshold (Kd = Kb/d)
        
        :param R_i: Distance from transmitter to scene center
        :param R_s: Distance from scene center to receiver
        
    """
    def __init__(self, k0, x, y, pol_i, pol_s,
                 theta_i, theta_s, dx, dy,
                 wind_U, wind_dir, wind_fetch,
                 sw_spec_func, sw_spread_func, d):

        # Save parameters
        self.wind_U = wind_U
        self.wind_dir = wind_dir
        self.wind_fetch = wind_fetch
        self.dx = dx
        self.dy = dy
        self.k0 = k0
        self.d = d
        self.pol_i = pol_i
        self.pol_s = pol_s
        self.theta_i = theta_i
        self.theta_s = theta_s
        self.sw_spec_func = sw_spec_func
        self.sw_dir_func = sw_spread_func
        self.shape = [y.shape[0], x.shape[0], 3]
        # Initialize Romeiser Interpolator (to increase speed!)
        # TODO: Initialize interpolator ONLY in Kb region
        #use always interpolator
        #if self.sw_spec_func == 'romeiser97':
        k = np.arange(1, 2*k0, 0.1)
        spectrum = self.interpolator = spec.models[self.sw_spec_func](k, self.wind_U, self.wind_fetch)
        #self.spec_interpolator = interpolate.InterpolatedUnivariateSpline(k, spectrum, k=1)
        self.spec_interpolator = interpolate.interp1d(k, spectrum,
                                                      bounds_error=False,
                                                      fill_value=1  )
        if (self.sw_dir_func == 'none') and (self.sw_spec_func == 'elfouhaily'):
            raise Exception('Elfouhaily spectrum requires a direction function')

    def rcs(self, phi_i, phi_s, Diffx, Diffy, monostatic = True):

        """ Calculates E.M. field

           
            :param diffx: Surface X slope
            :param diffy: Surface Y slope
            :param Dz: Surface height field
            :param Diffx: Space first derivatives (X slopes)
            :param Diffy: Space first derivatives (Y slopes)
            :param Diffxx: Space second derivatives (XX)
            :param Diffyy: Space second derivatives (YY)
            :param Diffxy: Space second derivatives (XY)
            :param monostatic: True for mono geometry

        """

        ### CACHE ###
#        sin_theta_i = np.sin(theta_i).reshape((1, theta_i.size))
         
       
        sin_theta_i = np.sin(self.theta_i)
        cos_theta_i = np.cos(self.theta_i)
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
#        if monostatic:
#            sin_theta_s = sin_theta_i
#            cos_theta_s = cos_theta_i
#            sin_phi_s = sin_phi_i
#            cos_phi_s = cos_phi_i
#            h_s = h_i
#            v_s = v_i
#        else:
        sin_theta_s = np.sin(self.theta_s)
        cos_theta_s = np.cos(self.theta_s)
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
        ####################################################### 
            
          ### VECTORS ###
        # Position (r) - Update heights
#        self.r[:, :, 2] = Dz
        
        # Polarization vectors (H, V)
        
#        a_i = h_i if pol_i == 'h' else v_i
#        a_s = h_s if pol_s == 'h' else v_s
        
        # Surface normal (n)
        n = np.empty(self.shape)
        n_norm = np.sqrt(Diffx**2. + Diffy**2. + 1.)
        n[:, :, 0] = -Diffx/n_norm
        n[:, :, 1] = -Diffy/n_norm
        n[:, :, 2] = 1./n_norm
        
        # Incidence direction (n_i)
        n_i = np.empty(self.shape)
        n_i[:, :, 0] = sin_theta_i*cos_phi_i
        n_i[:, :, 1] = sin_theta_i*sin_phi_i
        n_i[:, :, 2] = -cos_theta_i
        
#        t1 = np.linalg.norm(n_i)
        
        # Scattering direction (n_s)
        n_s = np.empty(self.shape)
        n_s[:, :, 0] = sin_theta_s * cos_phi_s
        n_s[:, :, 1] = sin_theta_s * sin_phi_s
        n_s[:, :, 2] = cos_theta_s
        
        # Scattering (q)
#        q = self.k0*(n_s - n_i)
        
        # Local frame of reference (t, d)
        t = np.cross(n, n_i)
        t /= np.sqrt(np.sum(t**2, axis=2)).reshape((self.shape[0], self.shape[1], 1))     ############### inja ham
        d = np.cross(t,n)
        ###############################
        cos_theta_il = -np.sum(n*n_i, axis=-1)
#        theta_il = np.arccos(cos_theta_il)
        sin_theta_il = np.sqrt(1 - cos_theta_il**2)   
#        sin_theta_il = sin_theta_il.reshape((1, theta_il.size))
#        cos_theta_il = np.cos(theta_il).reshape((1, theta_il.size))
        
        
        cos_phi_il = np.sum(n_i*d, axis=-1)/sin_theta_il                             #sin_theta_il
        sin_phi_il = np.nan_to_num(np.sqrt(1 - cos_phi_il**2))
#        phi_il = np.arccos(cos_phi_il)
        
        cos_theta_sl = np.sum(n*n_s, axis=-1)
        theta_sl = np.arccos(cos_theta_sl)
        sin_theta_sl = np.sqrt(1 - cos_theta_sl**2)   
#        sin_theta_sl = sin_theta_sl.reshape((1, theta_sl.size))
#        cos_theta_sl = np.cos(theta_sl).reshape((1, theta_sl.size))
        

        cos_phi_sl = np.sum(n_s*d, axis=-1)/sin_theta_sl                                   #sin_theta_sl be jaye  1
        sin_phi_sl = np.nan_to_num(np.sqrt(1 - cos_phi_sl**2))
#        phi_sl = np.arccos(cos_phi_sl)
        ###################################
        h_ii = np.empty(self.shape)
        h_ii[:, :, 0] = -sin_phi_il
        h_ii[:, :, 1] = cos_phi_il
        h_ii[:, :, 2] = 0.
        v_ii = np.empty(self.shape)
        v_ii[:, :, 0] = -cos_theta_il*cos_phi_il
        v_ii[:, :, 1] = -cos_theta_il*sin_phi_il
        v_ii[:, :, 2] = -sin_theta_il
############################  ino yadet bashe ke hazf shod ###########################        
#        if monostatic:
#            sin_theta_sl = sin_theta_il
#            cos_theta_sl = cos_theta_il
#            sin_phi_sl = sin_phi_il
#            cos_phi_sl = cos_phi_il
#            h_ss = h_ii
#            v_ss = v_ii
#        else:
#        sin_theta_sl = np.sin(theta_sl).reshape((1, theta_i.size))
#        cos_theta_sl = np.cos(theta_sl).reshape((1, theta_i.size))
###############################################################################################        
        h_ss = np.empty(self.shape)
        h_ss[:, :, 0] = -sin_phi_sl
        h_ss[:, :, 1] = cos_phi_sl
        h_ss[:, :, 2] = 0.
        v_ss = np.empty(self.shape)
        v_ss[:, :, 0] = cos_theta_s*cos_phi_sl
        v_ss[:, :, 1] = cos_theta_s*sin_phi_sl
        v_ss[:, :, 2] = -sin_theta_sl
          ############################################
          
             # Bragg wave number magnitude and direction (7) (8)
#        cos_phi_si = np.cos(phi_sl - phi_il)
#        sin_phi_si = np.sin(phi_sl - phi_il)
        cos_phi_Dl = cos_phi_sl*cos_phi_il + sin_phi_sl*sin_phi_il
        sin_phi_Dl = sin_phi_sl*cos_phi_il - sin_phi_il*cos_phi_sl

        k_b_x = self.k0*(sin_theta_sl*cos_phi_Dl + sin_theta_il)   ##############n   inja masle hast - +
        k_b_y = self.k0*(sin_theta_sl*sin_phi_Dl )
        k_b = np.sqrt(k_b_x**2 + k_b_y**2)
#        k_b = np.where(k_b > (2 * self.k0), 0, k_b)    ##############   ichoooooooo
#        phi_b =  np.arctan(k_b_y / k_b_x) 
        phi_b = np.arctan((k_b_x * d[:,:, 1] + k_b_y * t[:,:, 1]) / (k_b_x * d[:,:, 0] + k_b_y * t[:,:, 0]))
        
        
        
        # Calculate folded spectrum
        k_inp = np.array([k_b, k_b])
        
        theta_inp = np.array([phi_b, phi_b + np.pi]) - self.wind_dir
        theta_inp = np.angle(np.exp(1j * theta_inp))
        #if self.sw_spec_func == 'romeiser97':
        spectrum_1D = self.spec_interpolator(k_b.flatten()).reshape((1,) +
                                                                    k_b.shape)
        #else:
        #    spectrum = spec.models[self.sw_spec_func](k_inp, self.U_10, self.fetch)
        spectrum = spectrum_1D 
        spectrum = (spectrum_1D / k_inp *
                    spread.models[self.sw_dir_func](k_inp, theta_inp,
                                                    self.wind_U,
                                                    self.wind_fetch))
        bad_ones = np.where(np.logical_or(k_b < self.k0 * self.d, cos_theta_il < 0, cos_theta_sl < 0))
#        bad_ones_n = np.where(np.logical_or(cos_theta_il < 0, cos_theta_sl < 0))
        #######################################################
        #polarization amplitudes coefficients
        A_hh = ((const.epsilon_sw - 1)*cos_phi_Dl/((cos_theta_il + np.sqrt(const.epsilon_sw -sin_theta_il**2))*
                (cos_theta_sl + np.sqrt(const.epsilon_sw -sin_theta_sl**2))))
                                                  
        A_vh = (sin_phi_Dl*(const.epsilon_sw-1)*np.sqrt(const.epsilon_sw -sin_theta_sl**2)/
                           ((cos_theta_il + np.sqrt(const.epsilon_sw -sin_theta_il**2))*
                            (const.epsilon_sw*cos_theta_sl + np.sqrt(const.epsilon_sw -sin_theta_sl**2))))
        A_hv = (sin_phi_Dl*(const.epsilon_sw-1)*np.sqrt(const.epsilon_sw -sin_theta_il**2)/
                           ((const.epsilon_sw*cos_theta_il + np.sqrt(const.epsilon_sw -sin_theta_il**2))*
                            (cos_theta_sl + np.sqrt(const.epsilon_sw -sin_theta_sl**2))))
        A_vv = ((const.epsilon_sw-1)*(const.epsilon_sw*sin_theta_il*sin_theta_sl - cos_phi_Dl*np.sqrt(const.epsilon_sw -sin_theta_il**2)*np.sqrt(const.epsilon_sw -sin_theta_sl**2))/
                ((const.epsilon_sw*cos_theta_il + np.sqrt(const.epsilon_sw -sin_theta_il**2))*
                 (const.epsilon_sw*cos_theta_sl + np.sqrt(const.epsilon_sw -sin_theta_sl**2))))
        #######################################################
        
        rcs_hhhv= 16*self.k0**3*cos_theta_il**2*cos_theta_sl**2*(A_hh*np.conj(A_hv)).real*spectrum
        rcs_hhvh= 16*self.k0**3*cos_theta_il**2*cos_theta_sl**2*(A_hh*np.conj(A_vh)).real*spectrum
        rcs_vvhh= 16*self.k0**3*cos_theta_il**2*cos_theta_sl**2*(A_vv*np.conj(A_hh)).real*spectrum
        rcs_hvvh= 16*self.k0**3*cos_theta_il**2*cos_theta_sl**2*(A_hv*np.conj(A_vh)).real*spectrum
        rcs_hvvv= 16*self.k0**3*cos_theta_il**2*cos_theta_sl**2*(A_hv*np.conj(A_vv)).real*spectrum
        rcs_vvvh= 16*self.k0**3*cos_theta_il**2*cos_theta_sl**2*(A_vv*np.conj(A_vh)).real*spectrum
        rcs_hh= 8*self.k0**3*cos_theta_il**2*cos_theta_sl**2*A_hh**2*spectrum
        rcs_vv= 8*self.k0**3*cos_theta_il**2*cos_theta_sl**2*A_vv**2*spectrum
        rcs_hv= 8*self.k0**3*cos_theta_il**2*cos_theta_sl**2*A_hv**2*spectrum
        rcs_vh= 8*self.k0**3*cos_theta_il**2*cos_theta_sl**2*A_vh**2*spectrum
        
        #######################################################
        rcs_avg_vv = np.zeros([self.shape[0], self.shape[1]], dtype=np.complex)
        rcs_avg_hv = np.zeros([self.shape[0], self.shape[1]], dtype=np.complex)
        rcs_avg_vh = np.zeros([self.shape[0], self.shape[1]], dtype=np.complex)
        rcs_avg_hh = np.zeros([self.shape[0], self.shape[1]], dtype=np.complex)
        #######################################################
        if self.pol_i == 'v':
            if self.pol_s == 'v':
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        rcs_avg_vv[i,j] =  (np.inner(v_s[i, j],v_ss[i, j])**2*np.inner(v_i[i, j],v_ii[i, j])**2*rcs_vv[0,i, j] + np.inner(v_s[i, j],v_ss[i, j])**2*np.inner(v_i[i, j],h_ii[i, j])**2*rcs_vh[0,i, j]  +    
                                   np.inner(v_s[i, j],h_ss[i, j])**2*np.inner(v_i[i, j],v_ii[i, j])**2*rcs_hv[0,i, j]  + np.inner(v_s[i, j],h_ss[i, j])**2*np.inner(v_i[i, j],h_ii[i, j])**2*rcs_hh[0,i, j] +
                                   np.inner(v_s[i, j],h_ss[i, j])**2*np.inner(v_i[i, j],v_ii[i, j])*np.inner(v_i[i, j],h_ii[i, j])*rcs_hhhv[0,i, j] +
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],h_ii[i, j])**2*rcs_hhvh[0,i, j] +
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],v_ii[i, j])*np.inner(v_i[i, j],h_ii[i, j])*rcs_vvhh[0,i, j]+
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],h_ii[i, j])*np.inner(v_i[i, j],v_ii[i, j])*rcs_hvvh[0,i, j]+
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],v_ii[i, j])**2*rcs_hvvv[0,i, j]+
                                   np.inner(v_s[i, j],v_ss[i, j])**2*np.inner(v_i[i, j],h_ii[i, j])*np.inner(v_i[i, j],v_ii[i, j])*rcs_vvvh[0,i, j])
                rcs_avg_vv[bad_ones] = 0
#                rcs_avg_vv[1][bad_ones] = 0
                rcs =  rcs_avg_vv * self.dx * self.dy 
                return rcs
            
            
            
            
            if self.pol_s == 'h':
                 for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        rcs_avg_hv[i,j] =  (np.inner(h_s[i, j],v_ss[i, j])**2*np.inner(v_i[i, j],v_ii[i, j])**2*rcs_vv[0,i, j] + np.inner(h_s[i, j],v_ss[i, j])**2*np.inner(v_i[i, j],h_ii[i, j])**2*rcs_vh[0,i, j]  +    
                                   np.inner(h_s[i, j],h_ss[i, j])**2*np.inner(v_i[i, j],v_ii[i, j])**2*rcs_hv[0,i, j]  + np.inner(h_s[i, j],h_ss[i, j])**2*np.inner(v_i[i, j],h_ii[i, j])**2*rcs_hh[0,i, j] +
                                   np.inner(h_s[i, j],h_ss[i, j])**2*np.inner(v_i[i, j],v_ii[i, j])*np.inner(v_i[i, j],h_ii[i, j])*rcs_hhhv[0,i, j] +
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],h_ii[i, j])**2*rcs_hhvh[0,i, j] +
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],v_ii[i, j])*np.inner(v_i[i, j],h_ii[i, j])*rcs_vvhh[0,i, j]+
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],h_ii[i, j])*np.inner(v_i[i, j],v_ii[i, j])*rcs_hvvh[0,i, j]+
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(v_i[i, j],v_ii[i, j])**2*rcs_hvvv[0,i, j]+
                                   np.inner(h_s[i, j],v_ss[i, j])**2*np.inner(v_i[i, j],h_ii[i, j])*np.inner(v_i[i, j],v_ii[i, j])*rcs_vvvh[0,i, j])
                 rcs_avg_hv[bad_ones] = 0
#                 rcs_avg_hv[0][bad_ones] = 0
                 rcs = rcs_avg_hv * self.dx * self.dy
                 return rcs
            
            if self.pol_s == 'v' or 'h':
                rcs = rcs_vv + rcs_hv
                return rcs
       
        
        if self.pol_i == 'h':
            if self.pol_s == 'h':
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        rcs_avg_hh[i,j] =  (np.inner(h_s[i, j],v_ss[i, j])**2*np.inner(h_i[i, j],v_ii[i, j])**2*rcs_vv[0,i, j] + np.inner(h_s[i, j],v_ss[i, j])**2*np.inner(h_i[i, j],h_ii[i, j])**2*rcs_vh[0,i, j]  +    
                                   np.inner(h_s[i, j],h_ss[i, j])**2*np.inner(h_i[i, j],v_ii[i, j])**2*rcs_hv[0,i, j]  + np.inner(h_s[i, j],h_ss[i, j])**2*np.inner(h_i[i, j],h_ii[i, j])**2*rcs_hh[0,i, j] +
                                   np.inner(h_s[i, j],h_ss[i, j])**2*np.inner(h_i[i, j],v_ii[i, j])*np.inner(h_i[i, j],h_ii[i, j])*rcs_hhhv[0,i, j] +
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],h_ii[i, j])**2*rcs_hhvh[0,i, j] +
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],v_ii[i, j])*np.inner(h_i[i, j],h_ii[i, j])*rcs_vvhh[0,i, j]+
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],h_ii[i, j])*np.inner(h_i[i, j],v_ii[i, j])*rcs_hvvh[0,i, j]+
                                   np.inner(h_s[i, j],v_ss[i, j])*np.inner(h_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],v_ii[i, j])**2*rcs_hvvv[0,i, j]+
                                   np.inner(h_s[i, j],v_ss[i, j])**2*np.inner(h_i[i, j],h_ii[i, j])*np.inner(h_i[i, j],v_ii[i, j])*rcs_vvvh[0,i, j])
                rcs_avg_hh[bad_ones] = 0
#                rcs_avg_hh[0][bad_ones] = 0
                rcs = rcs_avg_hh * self.dx * self.dy
                return rcs
        
            if self.pol_s == 'v':
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        rcs_avg_vh[i,j] =  (np.inner(v_s[i, j],v_ss[i, j])**2*np.inner(h_i[i, j],v_ii[i, j])**2*rcs_vv[0,i, j] + np.inner(v_s[i, j],v_ss[i, j])**2*np.inner(h_i[i, j],h_ii[i, j])**2*rcs_vh[0,i, j]  +    
                                   np.inner(v_s[i, j],h_ss[i, j])**2*np.inner(h_i[i, j],v_ii[i, j])**2*rcs_hv[0,i, j]  + np.inner(v_s[i, j],h_ss[i, j])**2*np.inner(h_i[i, j],h_ii[i, j])**2*rcs_hh[0,i, j] +
                                   np.inner(v_s[i, j],h_ss[i, j])**2*np.inner(h_i[i, j],v_ii[i, j])*np.inner(h_i[i, j],h_ii[i, j])*rcs_hhhv[0,i, j] +
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],h_ii[i, j])**2*rcs_hhvh[0,i, j] +
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],v_ii[i, j])*np.inner(h_i[i, j],h_ii[i, j])*rcs_vvhh[0,i, j]+
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],h_ii[i, j])*np.inner(h_i[i, j],v_ii[i, j])*rcs_hvvh[0,i, j]+
                                   np.inner(v_s[i, j],v_ss[i, j])*np.inner(v_s[i, j],h_ss[i, j])*np.inner(h_i[i, j],v_ii[i, j])**2*rcs_hvvv[0,i, j]+
                                   np.inner(v_s[i, j],v_ss[i, j])**2*np.inner(h_i[i, j],h_ii[i, j])*np.inner(h_i[i, j],v_ii[i, j])*rcs_vvvh[0,i, j])
                rcs_avg_vh[bad_ones] = 0
#                rcs_avg_vh[0][bad_ones] = 0
                rcs = rcs_avg_vh * self.dx * self.dy
                return rcs
    
            if self.pol_s == 'v' or 'h':
                rcs = rcs_hh + rcs_vh
                return rcs
        
        
        if self.pol_i == 'DP':
            if self.pol_s == 'h':
                rcs = rcs_hh + rcs_hv
                return rcs
            
            if self.pol_s == 'v':
                rcs = rcs_vv + rcs_vh
                return rcs
            
            if self.pol_s == 'v' or 'h':
                rcs = rcs_hh + rcs_vv + rcs_hv + rcs_vh
                return rcs
                
        
        
        
