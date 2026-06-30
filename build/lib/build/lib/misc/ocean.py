
from time import time

import numpy as np
from scipy import weave
from scipy.constants import g

from trampa import utils

from wavesimlib import spec
from wavesimlib import spread

class OceanSurface(object):
    """ Ocean surface simulator class 
    
        :param Lx: Scene X size (m)
        :param Ly: Scene Y size (m)
        :param dx: Scene X resolution (m)
        :param dy: Scene Y resolution (m)
        :param spec_model: Omnidirectional Spectrum model
        :param spread_model: Spreading function model
        :param wind_dir: Wind direction (rad)
        :param wind_fetch: Wind fetch
        :param wind_U: Wind speed (m/s)
        :param swell_enable: EN/DIS Swell
        :param swell_rms: Swell RMS (m)
        :param swell_dir: Swell direction (rad)
        :param swell_wl: Swell peak wavelength (m)
        :param swell_sigma_x: Swell deviation in X direction (rad/s)
        :param swell_sigma_y: Swell deviation in Y direction (rad/s)
        :param compute_wf: EN/DIS Wavefield computation
        :param compute_diff: EN/DIS Spatial 1st derivatives computation
        :param compute_diff2: EN/DIS Spatial 2nd derivatives computation
        :param compute_vel: EN/DIS Velocities computation
        :param compute_acc: EN/DIS Acceleration computation
        :param optimize_res: Automatically adjust resolution to have optimal matrix sizes
        :param fft_max_prime: Maximum prime factor allowed in matrix sizes
        :param min_wl: Minimum spectrum wavelength
        
        ..note::
            Thanks Paco Lopez-Dekker for the swell implementation idea
    """
    
    def __init__(self, Lx, Ly, dx, dy,
                 spec_model, spread_model,
                 wind_dir, wind_fetch, wind_U, 
                 swell_enable=False, swell_rms=0., swell_dir=0., swell_wl=0., swell_sigma_x=0., swell_sigma_y=0.,
                 compute_wf=True, compute_diff=False, compute_diff2=False, compute_vel=False, compute_acc=False,
                 optimize_res=True, fft_max_prime=3, min_wl=None):

        # Save some values
        self.swell_enable = swell_enable
        self.compute_wf = compute_wf
        self.compute_diff = compute_diff
        self.compute_diff2 = compute_diff2
        self.compute_vel = compute_vel
        self.compute_acc = compute_acc
        
        # Grid dimensions
        Nx = np.int(Lx/dx)
        Ny = np.int(Ly/dy)
        
        if optimize_res:
            Nx = utils.optimize_fftsize(Nx, fft_max_prime)
            Ny = utils.optimize_fftsize(Ny, fft_max_prime)
            
            dx = Lx/np.float(Nx)
            dy = Ly/np.float(Ny)

        # Maximum Kx, Ky (Sampling theorem, 2*pi/(2*res))
        if not min_wl:
            kmax_x = np.pi/dx
            kmax_y = np.pi/dy
        else:
            kmax_x = 2.*np.pi/min_wl
            kmax_y = 2.*np.pi/min_wl
            
            if (kmax_x > np.pi/dx) or (kmax_y > np.pi/dy):
                raise ValueError('Minimum wavelength is too small for the specified grid resolution')
        
        # Kx-Ky meshgrid (0:N/2, -N/2:-1)
        kx_o = np.linspace(-kmax_x, kmax_x, Nx)
        kx_s = np.empty(Nx)
        kx_s[:Nx/2], kx_s[Nx/2:] = kx_o[Nx/2:], kx_o[:Nx/2]
        ky_o = np.linspace(-kmax_y, kmax_y, Ny)
        ky_s = np.empty(Ny)
        ky_s[:Ny/2], ky_s[Ny/2:] = ky_o[Ny/2:], ky_o[:Ny/2]
        self.kx, self.ky = np.meshgrid(kx_s, ky_s)
        
        # Kx-Ky resolution
        kx_res = self.kx[0,1] - self.kx[0,0]
        ky_res = self.ky[1,0] - self.ky[0,0]
        
        # K-theta meshgrid (Polar, wind direction shifted)
        self.k = np.sqrt(self.kx**2 + self.ky**2)
        self.theta = np.arctan2(self.ky, self.kx) - wind_dir
        
        # omega (w(k)^2 = g*k)
        self.omega = np.sqrt(g*self.k)
        
        # Compute directional wave spectrum (1/k*S(k)*D(k,theta))
        if spec_model not in spec.models:
            raise NotImplementedError('%s spectrum function not implemented' % spec_model)
        if spread_model not in spread.models:
            raise NotImplementedError('%s spreading function not implemented' % spread_model)
        
        wave_spec = spec.models[spec_model](self.k, wind_U, wind_fetch)
        wave_spread = spread.models[spread_model](self.k, self.theta, wind_U, wind_fetch)
        wave_dirspec = (1./self.k)*wave_spec*wave_spread

        # Initialize coefficients & randomize
        self.wave_coefs = Nx*Ny*np.sqrt(kx_res*ky_res)*(np.random.normal(0, 1, size=[Nx, Ny]) + 
                                                    1j*np.random.normal(0, 1, size=[Nx, Ny])) * np.sqrt(wave_dirspec)
    
        # Swell
        if swell_enable:

            # Swell peak
            k_p = 2.*np.pi/swell_wl
            
            # Swell Kx-Ky meshgrid (Gaussian centered)
            kx_s = k_p + np.linspace(-2.*swell_sigma_x, 2.*swell_sigma_x, 8)
            ky_s = np.linspace(-2.*swell_sigma_y, 2.*swell_sigma_y, 8)
            kx_m, ky_m = np.meshgrid(kx_s, ky_s)
            self.swell_kx = kx_m*np.cos(swell_dir) - ky_m*np.sin(swell_dir)
            self.swell_ky = kx_m*np.sin(swell_dir) + ky_m*np.cos(swell_dir)
            
            kx_res = kx_s[1] - kx_s[0]
            ky_res = ky_s[1] - ky_s[0]
            
            # Swell k, omega (w^2 = gk)
            self.swell_k = np.sqrt(self.swell_kx**2 + self.swell_ky**2)
            self.swell_omega = np.sqrt(g*self.swell_k)
            
            # Spectrum coefficients
            swell_spec = swell_rms**2./(16.*2.*np.pi*swell_sigma_x*swell_sigma_y)*np.exp(-0.5*(((self.swell_kx - k_p*np.cos(swell_dir))/swell_sigma_x)**2 + 
                                                                                           ((self.swell_ky - k_p*np.sin(swell_dir))/swell_sigma_y)**2))
            
            self.swell_coefs = np.sqrt(2.*swell_spec*kx_res*ky_res)#*np.exp(1j*2.*np.pi*np.random.uniform(0., 1., size=swell_spec.shape))
            
            # Create cache for sin/cos in swell IDFT
            x = np.linspace(0., Lx, Nx)
            y = np.linspace(0., Ly, Ny)
            self.swell_cos_kx = np.empty([x.shape[0], self.swell_kx.shape[0], self.swell_ky.shape[0]])
            self.swell_cos_ky = np.empty([y.shape[0], self.swell_kx.shape[0], self.swell_ky.shape[0]])
            self.swell_sin_kx = np.empty_like(self.swell_cos_kx)
            self.swell_sin_ky = np.empty_like(self.swell_cos_ky)
            for i in np.arange(x.shape[0]):
                self.swell_cos_kx[i] = np.cos(x[i]*self.swell_kx)
                self.swell_sin_kx[i] = np.sin(x[i]*self.swell_kx)
            for j in np.arange(y.shape[0]):
                self.swell_cos_ky[j] = np.cos(y[j]*self.swell_ky)
                self.swell_sin_ky[j] = np.sin(y[j]*self.swell_ky)
            
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, value):
        
        self._t = value
                
        # Propagate
        wave_coefs_phased = self.wave_coefs*np.exp(-1j*self.omega*self._t)
        if self.swell_enable:
            swell_coefs_phased = self.swell_coefs*np.exp(-1j*self.swell_omega*self._t)
        
        # HORIZ. DISPL. & HEIGHT FIELD (Dx, Dy, Dz)
        if self.compute_wf:
            #self.Dx = np.real(np.fft.ifft2(1j*self.kx/self.k*wave_coefs_phased))
            #self.Dy = np.real(np.fft.ifft2(1j*self.ky/self.k*wave_coefs_phased))
            #self.Dz = np.real(np.fft.ifft2(wave_coefs_phased))
            if self.swell_enable:
                self.Dx = self.__swell_idft(1j*self.swell_kx/self.swell_k*swell_coefs_phased)
                self.Dy = self.__swell_idft(1j*self.swell_ky/self.swell_k*swell_coefs_phased)
                self.Dz = self.__swell_idft(swell_coefs_phased)
 
        # FIRST SPATIAL DERIVATIVES - SLOPES (Diffx, Diffy)
        if self.compute_diff:
            self.Diffx = np.real(np.fft.ifft2(1j*self.kx*wave_coefs_phased))
            self.Diffy = np.real(np.fft.ifft2(1j*self.ky*wave_coefs_phased))
            if self.swell_enable:
                self.Diffx += self.__swell_idft(1j*self.swell_kx*swell_coefs_phased)
                self.Diffy += self.__swell_idft(1j*self.swell_ky*swell_coefs_phased)
                       
        # SECOND SPATIAL DERIVATIVES (Diffxx, Diffyy, Diffxy)
        if self.compute_diff2:
            self.Diffxx = np.real(np.fft.ifft2(-self.kx**2*wave_coefs_phased))
            self.Diffyy = np.real(np.fft.ifft2(-self.ky**2*wave_coefs_phased))
            self.Diffxy = np.real(np.fft.ifft2(-self.kx*self.ky*wave_coefs_phased))
            if self.swell_enable:
                self.Diffxx += self.__swell_idft(-self.swell_kx*swell_coefs_phased)
                self.Diffyy += self.__swell_idft(-self.swell_ky*swell_coefs_phased)
                self.Diffxy += self.__swell_idft(-self.swell_kx*self.swell_ky*swell_coefs_phased)
        
        # FIRST TIME DERIVATIVES - VELOCITY (Vx, Vy, Vz)
        if self.compute_vel:
            wave_coefs_diff_t_phased = -1j*self.omega*wave_coefs_phased
            self.Vx = np.real(np.fft.ifft2(1j*self.kx/self.k*wave_coefs_diff_t_phased))
            self.Vy = np.real(np.fft.ifft2(1j*self.ky/self.k*wave_coefs_diff_t_phased))
            self.Vz = np.real(np.fft.ifft2(wave_coefs_diff_t_phased))
            if self.swell_enable:
                swell_coefs_diff_t_phased = -1j*self.swell_omega*swell_coefs_phased
                self.Vx += self.__swell_idft(1j*self.swell_kx/self.swell_k*swell_coefs_diff_t_phased)
                self.Vy += self.__swell_idft(1j*self.swell_ky/self.swell_k*swell_coefs_diff_t_phased)
                self.Vz += self.__swell_idft(swell_coefs_diff_t_phased)
        
        # SECOND TIME DERIVATIVES - ACCELERATION (Ax, Ay, Az)
        if self.compute_acc:
            wave_coefs_diff2_t_phased = -self.omega**2.*wave_coefs_phased
            self.Ax = np.real(np.fft.ifft2(1j*self.kx/self.k*wave_coefs_diff2_t_phased))
            self.Ay = np.real(np.fft.ifft2(1j*self.ky/self.k*wave_coefs_diff2_t_phased))
            self.Az = np.real(np.fft.ifft2(wave_coefs_diff2_t_phased))
            if self.swell_enable:
                swell_coefs_diff2_t_phased = -self.swell_omega**2.*swell_coefs_phased
                self.Ax += self.__swell_idft(1j*self.swell_kx/self.swell_k*swell_coefs_diff2_t_phased)
                self.Ay += self.__swell_idft(1j*self.swell_ky/self.swell_k*swell_coefs_diff2_t_phased)
                self.Az += self.__swell_idft(swell_coefs_diff2_t_phased)
    
    def swell_idft(self, coefs):
        return self.__swell_idft(coefs)
                
    def __swell_idft(self, coefs):
        """ Performs Inverse Fourier Transform 
            Transform is performed given two *independent*
            sampled spaces (XY-Space, KxKy-Space)
            
            :param coefs: Complex Fourier coefficients
            
            .. note::
                Only real part is computed (imag is not needed by this module)
                Requires sin/cos to be initialized!
        """    
        
        code = """ 
        int i, j, m, n;
        float re_val;

        // Loop through X-Y space
        for(i=0; i<Ncos_kx[0]; i++)
        {
            for(j=0; j<Ncos_ky[0]; j++)
            {
                re_val = 0;
                // Loop through Kx-Ky space
                for(m=0; m<Ncos_kx[1]; m++)
                {
                    for(n=0; n<Ncos_ky[1]; n++)
                    {
                        re_val += COEFS_R2(m,n)*(COS_KX3(i,m,n)*COS_KY3(j,m,n) - SIN_KX3(i,m,n)*SIN_KY3(j,m,n)) - \
                                  COEFS_I2(m,n)*(SIN_KX3(i,m,n)*COS_KY3(j,m,n) + COS_KX3(i,m,n)*SIN_KY3(j,m,n));
                    }
                }
                OUT2(i,j) = re_val;
            }
        }
        """
        
        # Setup variables
        coefs_r = coefs.real
        coefs_i = coefs.imag
        cos_kx = self.swell_cos_kx
        sin_kx = self.swell_sin_kx
        cos_ky = self.swell_cos_ky
        sin_ky = self.swell_sin_ky
        out = np.empty([self.swell_cos_kx.shape[0], self.swell_cos_ky.shape[0]])
        
        weave.inline(code, ['coefs_r', 'coefs_i', 'cos_kx', 'sin_kx','cos_ky', 'sin_ky', 'out'])
    
        return out