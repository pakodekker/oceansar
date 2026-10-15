
import numpy as np
import scipy as sp
from scipy import linalg
import types


from oceansar import ocs_io as wsio
from oceansar import spec
from oceansar import spread
from drama import utils
from oceansar import constants as const



class OceanSurface(object):
    """ Ocean Surface class

        This class is used to simulate ocean surfaces.
        Given an initial description of the surface, or
        an state file (initialized surface file) it can
        compute many surface properties (height field,
        slopes, velocities...) for any time instant.
    """

    def __init__(self):
        self.initialized = False


    def init(self, Lx, Ly, dx, dy, cutoff_wl,
             spec_model, spread_model,
             wind_dir, wind_fetch, wind_U,
             current_mag, current_dir,
             dir_swell_dir=0, freq_r=100, sigf=1, sigs=1, Hs=2,
             swell_dir_enable=False,
             swell_enable=False, swell_ampl=0., swell_dir=0., swell_wl=0.,
             compute=[], opt_res=True, fft_max_prime=2, choppy_enable=False,
             dirspectrum_func=None, depth=None, dir_swell_spec=None,
             workers=4, rng=None):

        """Initialize a random ocean Fourier realization and allocate requested fields.

        Parameters
        ----------
        Lx, Ly : float
            Scene dimensions in metres, along x (range) and y (azimuth).
        dx, dy : float
            Requested grid spacing in metres. FFT-size optimization can adjust
            the actual spacing while preserving Lx and Ly.
        cutoff_wl : float, 'auto' or None
            Spectral cutoff wavelength in metres. None or 'auto' leaves the
            spectrum limited by the grid sampling.
        spec_model, spread_model : str
            Keys in the OceanSAR omnidirectional-spectrum and spreading registries.
            Used when dirspectrum_func is not supplied.
        wind_dir : float
            Wind direction in radians in the surface x/y coordinate system.
        wind_fetch : float
            Wind fetch in metres.
        wind_U : float
            Wind speed in m/s; the current is subtracted to obtain relative wind.
        current_mag : float
            Surface-current speed in m/s.
        current_dir : float
            Surface-current direction in radians.
        dir_swell_dir : float, optional
            Mean directional-swell direction passed to dir_swell_spec. The bundled
            directional-swell model expects degrees.
        freq_r, sigf : float, optional
            Directional-swell peak frequency and frequency spread in Hz, using
            the convention of the supplied dir_swell_spec callable.
        sigs : float, optional
            Directional-swell angular spread; degrees for the bundled model.
        Hs : float, optional
            Directional-swell significant wave height in metres.
        swell_dir_enable : bool, optional
            Add a random directional-swell spectrum using dir_swell_spec.
        swell_enable : bool, optional
            Add a monochromatic swell with random initial phase.
        swell_ampl : float, optional
            Monochromatic-swell displacement amplitude in metres.
        swell_dir : float, optional
            Monochromatic-swell direction in radians.
        swell_wl : float, optional
            Monochromatic-swell wavelength in metres; must be positive when enabled.
        compute : sequence of str, optional
            Fields to allocate: 'D' displacements (m), 'Diff' slopes (dimensionless),
            'Diff2' second spatial derivatives (1/m), 'V' velocities (m/s), 'A'
            accelerations (m/s**2), or 'hMTF' hydrodynamic modulation (dimensionless).
            'Diff2' also enables 'Diff'.
        opt_res : bool, optional
            Adjust grid dimensions to efficient FFT sizes; default True.
        fft_max_prime : int, optional
            Largest allowed prime factor in optimized FFT sizes; default 2.
        choppy_enable : bool, optional
            Include choppy-wave corrections when evaluating surface fields.
        dirspectrum_func : callable, optional
            External directional spectrum callable accepting kx, ky arrays in
            rad/m and returning spectral density in m**4 (height variance per
            two-dimensional wavenumber area). Replaces the wind-spectrum models.
        depth : float, optional
            Water depth in metres for finite-depth dispersion. None uses deep water.
        dir_swell_spec : callable, optional
            Required when swell_dir_enable=True. Called with
            (k, theta, dir_swell_dir, freq_r, sigf, sigs, Hs), with k in rad/m and
            theta in radians, returning the directional swell spectrum before
            conversion from polar to Cartesian wavenumber density.
        workers : int, optional
            Worker count passed to SciPy FFT operations; default 4.
        rng : NumPy-compatible random generator, optional
            Source of normal and uniform draws. None uses numpy.random.

        Returns
        -------
        None
            Populate this object and mark it initialized. wave_coefs and allocated
            surface fields have shape (Ny, Nx); x/y axes are in metres and kx/ky
            grids in rad/m. Set surface.t in seconds to evaluate requested fields.
        """

        if rng is None:
            rng = np.random

        ## INITIALIZE CLASS VARIABLES
        # Save surface properties
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.wind_dir = wind_dir
        self.wind_fetch = wind_fetch
        self.wind_U = wind_U
        self.current_mag = current_mag
        self.current_dir = current_dir
        self.swell_enable = swell_enable
        self.swell_ampl = swell_ampl
        self.swell_dir = swell_dir
        self.swell_wl = swell_wl
        self.compute = list(compute)
        if 'Diff2' in self.compute and 'Diff' not in self.compute:
            self.compute.append('Diff')
        self.choppy_enable = choppy_enable
        self.dir_swell_dir = dir_swell_dir
        self.freq_r = freq_r
        self.sigf = sigf
        self.sigs = sigs
        self.Hs = Hs
        self.workers = workers
        self.depth = depth
        ## INITIALIZE MESHGRIDS, SPECTRUM, ETC.
        # Grid dimensions
        self.Nx = int(self.Lx/self.dx)
        self.Ny = int(self.Ly/self.dy)

        if opt_res:
            self.Nx = int(utils.optimize_fftsize(self.Nx, fft_max_prime))
            self.Ny = int(utils.optimize_fftsize(self.Ny, fft_max_prime))

            self.dx = np.float32(self.Lx/float(self.Nx))
            self.dy = np.float32(self.Ly/float(self.Ny))

        # X-Y vector
        self.x = np.linspace(-self.Lx/2., self.Lx/2., self.Nx, dtype=np.float32)
        self.y = np.linspace(-self.Ly/2., self.Ly/2., self.Ny, dtype=np.float32)

        # Currents
        self.current = (self.current_mag * np.array([np.cos(self.current_dir),
                                                     np.sin(self.current_dir)])).astype(np.float32)
        U_eff_vec = (self.wind_U * np.array([np.cos(self.wind_dir),
                                             np.sin(self.wind_dir)]) -
                     self.current).astype(np.float32)
        self.wind_U_eff = linalg.norm(U_eff_vec)
        self.wind_dir_eff = np.arctan2(U_eff_vec[1], U_eff_vec[0])

        # Maximum Kx, Ky (Sampling theorem, 2*pi/(2*res))
        kmax_x = np.float32(np.pi/self.dx)
        kmax_y = np.float32(np.pi/self.dy)

        # Kx-Ky meshgrid (0:N/2, -N/2:-1)
        #kx_o = np.linspace(-kmax_x, kmax_x, self.Nx)
        kx_s = (2*np.pi*sp.fft.fftfreq(self.Nx, self.dx)).astype(np.float32)
        #ky_o = np.linspace(-kmax_y, kmax_y, self.Ny)
        ky_s = (2*np.pi*sp.fft.fftfreq(self.Ny, self.dy)).astype(np.float32)
        self.kx, self.ky = np.meshgrid(kx_s, ky_s)

        # Kx-Ky resolution
        self.kx_res = kx_res = self.kx[0, 1] - self.kx[0, 0]
        self.ky_res = ky_res = self.ky[1, 0] - self.ky[0, 0]

        # K-theta meshgrid (Polar, wind direction shifted)
        self.k = np.sqrt(self.kx**2 + self.ky**2)
        good_k = np.where(self.k > np.min(np.array([kx_res, ky_res])) / 2.0)
        self.kxn = np.zeros_like(self.kx, dtype=np.float32)
        self.kyn = np.zeros_like(self.kx, dtype=np.float32)
        self.kxn[good_k] = self.kx[good_k] / self.k[good_k]
        self.kyn[good_k] = self.ky[good_k] / self.k[good_k]
        self.kinv = np.zeros(self.k.shape, dtype=np.float32)
        self.kinv[good_k] = 1./self.k[good_k]
        #self.theta = np.arctan2(self.ky, self.kx) - self.wind_dir_eff
        self.theta = np.angle(np.exp(1j * (np.arctan2(self.ky, self.kx) -
                                           self.wind_dir_eff))).astype(np.float32)
        # for directional swell
        self.theta_swell = np.angle(np.exp(1j * (np.arctan2(self.ky, self.kx)))).astype(np.float32)
        # omega (Deep water: w(k)^2 = g*k)
        if depth is None:
            self.omega = np.sqrt(np.float32(const.g) * self.k)
        else:
            self.omega = np.sqrt(np.float32(const.g) * self.k * np.tanh(self.k * np.float32(depth)))

        # Compute directional wave spectrum (1/k*S(k)*D(k,theta))
        if callable(dirspectrum_func):
            self.wave_dirspec = np.zeros(self.k.shape, dtype=np.float32)
            self.wave_dirspec[good_k] = dirspectrum_func(self.kx[good_k], self.ky[good_k])
        else:
            if spec_model not in spec.models:
                raise NotImplementedError('%s spectrum function not implemented' % spec_model)
            if spread_model not in spread.models:
                raise NotImplementedError('%s spreading function not implemented' % spread_model)
            wave_spec = np.zeros(self.k.shape, dtype=np.float32)
            wave_spec[good_k] = spec.models[spec_model](self.k[good_k],
                                                        self.wind_U_eff,
                                                        self.wind_fetch)
            wave_spread = np.zeros(self.k.shape, dtype=np.float32)
            wave_spread[good_k] = spread.models[spread_model](self.k[good_k],
                                                              self.theta[good_k],
                                                              self.wind_U_eff,
                                                              self.wind_fetch)

            self.wave_dirspec = (self.kinv) * wave_spec * wave_spread
        # PLD May 2025
        self.rmss_x = np.sqrt(np.sum(self.wave_dirspec * self.kx**2 * self.kx_res*ky_res))
        self.rmss_y = np.sqrt(np.sum(self.wave_dirspec * self.ky**2 * self.kx_res*ky_res))
        print("rmssx: %f; rmssy %f" % (self.rmss_x,  self.rmss_y))
        # Filter if cutoff is imposed
        if cutoff_wl and (cutoff_wl != 'auto'):
            kco_x = 2.*np.pi/cutoff_wl
            kco_y = 2.*np.pi/cutoff_wl

            if (kco_x > np.pi/self.dx) or (kco_y > np.pi/self.dy):
                raise ValueError('Cutoff wavelength is too small for the specified grid resolution')

            k_f = np.zeros([self.Ny, self.Nx], dtype=np.float32)
            ky_sample = np.ceil((self.Ny/2)*kco_y/kmax_y)
            kx_sample = np.ceil((self.Nx/2)*kco_x/kmax_x)
            k_f[(k_f.shape[0]/2 - ky_sample):(k_f.shape[0]/2 + ky_sample),
                (k_f.shape[1]/2 - kx_sample):(k_f.shape[1]/2 + kx_sample)] = 1.

            self.wave_dirspec *= sp.fft.fftshift(k_f)

        # Complex Gaussian to randomize spectrum coefficients
        random_cg = (1./np.sqrt(2) * (rng.normal(0., 1., size=[self.Ny, self.Nx]) +
                                      1j * rng.normal(0., 1., size=[self.Ny, self.Nx]))).astype(np.complex64)

        # Swell spectrum
        # Swell directional spectrum
        if swell_dir_enable:
            swell_dirspec = np.zeros(self.k.shape, dtype=np.float32)
            swell_dirspec[good_k] = dir_swell_spec(self.k[good_k],
                                                   self.theta_swell[good_k],
                                                   self.dir_swell_dir,
                                                   self.freq_r, self.sigf,
                                                   self.sigs, self.Hs)
            self.swell_dirspec = (self.kinv) * swell_dirspec
            # Complex Gaussian to randomize spectrum coefficients
            random_swell = (1./np.sqrt(2) * (rng.normal(0., 1., size=[self.Ny, self.Nx]) +
                                          1j * rng.normal(0., 1., size=[self.Ny, self.Nx]))).astype(np.complex64)
        # Swell spectrum (monochromatic)
        if self.swell_enable:
            # Swell K, Kx, Ky, omega
            x, y = np.meshgrid(self.x, self.y)
            self.swell_k = np.float32(2.*np.pi/self.swell_wl)
            self.swell_kx = np.float32(self.swell_k*np.cos(self.swell_dir))
            self.swell_ky = np.float32(self.swell_k*np.sin(self.swell_dir))
            if depth is None:
                self.swell_omega = np.float32(np.sqrt(self.swell_k * const.g))
            else:
                self.swell_omega = np.float32(np.sqrt(self.swell_k*const.g * np.tanh(self.swell_k * depth)))
            # Swell initial phase (random)
            self.swell_ph0 = np.float32(rng.uniform(0., 2.*np.pi))
            # Swell in complex domain (monochromatic)
            self.swell_exp = (self.swell_ampl *
                              np.exp(1j*(self.swell_k*(np.cos(self.swell_dir)*x +
                                                       np.sin(self.swell_dir)*y) + self.swell_ph0))).astype(np.complex64)

        # Initialize coefficients
        self.wave_coefs = (self.Nx*self.Ny*np.sqrt(2.*self.wave_dirspec*kx_res*ky_res)*random_cg).astype(np.complex64)
        if swell_dir_enable:
            self.swell_coefs = (self.Nx*self.Ny*np.sqrt(2.*self.swell_dirspec*kx_res*ky_res)*random_swell).astype(np.complex64)
            self.wave_coefs = self.wave_coefs + self.swell_coefs
        # Allocate memory & mark as initialized
        self.__allocate()
        self.initialized = True

    def load(self, state_file, compute=[]):

        """Initialize this surface from saved native-complex Fourier initial conditions.

        Parameters
        ----------
        state_file : str or path-like
            OceanStateFile in the current complex-v1 HDF5 format. Contains full
            complex wave coefficients, 1D wavenumber axes and initialization
            metadata, including depth, RMS slopes and monochromatic swell phase.
            Older split-component files must be regenerated.
        compute : sequence of str, optional
            Fields to allocate: 'D' for displacements in m, 'Diff' for dimensionless
            slopes, 'Diff2' for second spatial derivatives in 1/m, 'V' for velocity
            in m/s, 'A' for acceleration in m/s**2, and 'hMTF' for the dimensionless
            hydrodynamic modulation. 'Diff2' also enables 'Diff'.

        Returns
        -------
        None
            Populate this object's grid, coefficients and model parameters and
            mark it initialized. Field arrays have shape (Ny, Nx). Set surface.t
            in seconds after loading to evaluate the requested fields at that time;
            loading does not itself compute a surface snapshot.
        """

        # Open file
        try:
            state = wsio.OceanStateFile(state_file, 'r')
        except RuntimeError:
            raise RuntimeError('Error loading OTGI File')

        # Load content
        self.Nx = int(state.get('Nx'))
        self.Ny = int(state.get('Ny'))
        self.Lx = state.get('Lx')
        self.Ly = state.get('Ly')
        self.dx = state.get('dx')
        self.dy = state.get('dy')
        self.workers = int(state.get('workers'))
        depth = float(state.get('water_depth'))
        self.depth = depth if np.isfinite(depth) else None
        self.rmss_x = state.get('rmss_x')
        self.rmss_y = state.get('rmss_y')
        self.wind_dir = np.deg2rad(state.get('wind_dir'))
        self.wind_dir_eff = np.deg2rad(state.get('wind_dir_eff'))
        self.wind_fetch = state.get('wind_fetch')
        self.wind_U = state.get('wind_U')
        self.wind_U_eff = state.get('wind_U_eff')
        self.current_mag = state.get('current_mag')
        self.current_dir = np.deg2rad(state.get('current_dir'))
        print("current_mag: %f; current_dir: %f" % (self.current_mag, np.rad2deg(self.current_dir)))
        self.swell_enable = True if state.get('swell_enable') == 1 else False
        self.swell_ampl = state.get('swell_ampl')
        self.swell_dir = np.deg2rad(state.get('swell_dir'))
        self.swell_wl = state.get('swell_wl')
        self.swell_ph0 = state.get('swell_ph0')
        self.kx, self.ky = np.meshgrid(state.get('kx'), state.get('ky'))
        self.wave_coefs = state.get('wave_coefs')
        self.choppy_enable = True if state.get('choppy_enable') == 1 else False
        

        state.close()
        self.k = np.sqrt(self.kx**2 + self.ky**2)
        self.kx_res = kx_res = self.kx[0, 1] - self.kx[0, 0]
        self.ky_res = ky_res = self.ky[1, 0] - self.ky[0, 0]

        # K-theta meshgrid (Polar, wind direction shifted)

        good_k = np.where(self.k > np.min(np.array([kx_res, ky_res]))/2.0)
        self.kxn = np.zeros_like(self.kx)
        self.kyn = np.zeros_like(self.kx)
        self.kxn[good_k] = self.kx[good_k] / self.k[good_k]
        self.kyn[good_k] = self.ky[good_k] / self.k[good_k]

        self.kinv = np.zeros(self.k.shape)
        self.kinv[good_k] = 1./self.k[good_k]
        ## Calculate derivated parameters
        # Current
        self.current = self.current_mag*np.array([np.cos(self.current_dir), np.sin(self.current_dir)])
        #print("current vector: %f, %f" % (self.current[0], self.current[1]))
        # X-Y vector
        self.x = np.linspace(-self.Lx/2, self.Lx/2., self.Nx, dtype=np.float32)
        self.y = np.linspace(-self.Ly/2., self.Ly/2., self.Ny, dtype=np.float32)

        # K-theta meshgrid (Polar, wind direction shifted)
        self.k = np.sqrt(self.kx**2 + self.ky**2)
        self.theta = np.angle(np.exp(1j * (np.arctan2(self.ky, self.kx) -
                                           self.wind_dir_eff)))

        # omega (Deep water: w(k)^2 = g*k)
        self.omega = np.sqrt(np.float32(const.g)*self.k)
        if self.depth is not None:
            self.omega = np.sqrt(np.float32(const.g)*self.k
                                 * np.tanh(self.k*np.float32(self.depth)))

        # Swell (NOTE: Maybe X-Y should be directly a meshgrid...)
        x, y = np.meshgrid(self.x, self.y)
        self.swell_k = np.float32(2.*np.pi/self.swell_wl)
        self.swell_kx = np.float32(self.swell_k*np.cos(self.swell_dir))
        self.swell_ky = np.float32(self.swell_k*np.sin(self.swell_dir))
        self.swell_omega = np.float32(np.sqrt(self.swell_k*const.g))
        if self.depth is not None:
            self.swell_omega = np.float32(np.sqrt(self.swell_k*const.g
                                                 * np.tanh(self.swell_k*self.depth)))
        self.swell_exp = (self.swell_ampl*np.exp(1j*(self.swell_k*(np.cos(self.swell_dir)*x +
                                                                 np.sin(self.swell_dir)*y) + self.swell_ph0))).astype(np.complex64)

        # Set compute, allocate memory & mark as initialized
        self.compute = list(compute)
        if 'Diff2' in self.compute and 'Diff' not in self.compute:
            self.compute.append('Diff')

        self.__allocate()
        self.initialized = True

    def save(self, state_file):
        """Save the initialized surface's Fourier realization and model parameters.

        Parameters
        ----------
        state_file : str or path-like
            Destination HDF5-backed OceanStateFile, overwritten if it exists.
            Stores full native-complex wave_coefs with shape (Ny, Nx), 1D kx/ky
            axes in rad/m and scalar initialization metadata. Coefficient dtype
            is preserved; no Hermitian symmetry is assumed.

        Returns
        -------
        None
            Write initial conditions for later load and propagation. Time-dependent
            surface rasters are excluded; depth, RMS slopes and monochromatic
            swell phase are retained.

        Raises
        ------
        Exception
            The surface has not been initialized.
        """

        if not self.initialized:
            raise Exception('Surface not initialized')

        # Create file
        state = wsio.OceanStateFile(state_file, 'w', [self.Ny, self.Nx])

        # Save content
        state.set('workers', self.workers)
        state.set('water_depth', self.depth if self.depth is not None else np.inf)
        state.set('rmss_x', self.rmss_x)
        state.set('rmss_y', self.rmss_y)
        state.set('Nx', self.Nx)
        state.set('Ny', self.Ny)
        state.set('Lx', self.Lx)
        state.set('Ly', self.Ly)
        state.set('dx', self.dx)
        state.set('dy', self.dy)
        state.set('wind_dir', np.rad2deg(self.wind_dir))
        state.set('wind_dir_eff', np.rad2deg(self.wind_dir_eff))
        state.set('wind_fetch', self.wind_fetch)
        state.set('wind_U', self.wind_U)
        state.set('wind_U_eff', self.wind_U_eff)
        state.set('current_mag', self.current_mag)
        state.set('current_dir', np.rad2deg(self.current_dir))
        state.set('swell_enable', 1 if self.swell_enable else 0)
        state.set('swell_ampl', self.swell_ampl)
        state.set('swell_dir', np.rad2deg(self.swell_dir))
        state.set('swell_wl', self.swell_wl)
        state.set('swell_ph0', self.swell_ph0 if self.swell_enable else 0.)
        state.set('kx', self.kx[0, :])
        state.set('ky', self.ky[:, 0])
        state.set('wave_coefs', self.wave_coefs)
        state.set('choppy_enable', 1 if self.choppy_enable else 0)

        state.close()

    def __allocate(self):
        """ Allocates surface properties arrays
            so contiguous memory is ensured (needed by Balancer)

            .. note::
                Internal use only
        """

        # Indices used to pack an arbitrary complex spectrum into the
        # Hermitian half-spectrum consumed by irfft2.
        self._neg_y = (-np.arange(self.Ny)) % self.Ny
        self._rfft_x = np.arange(self.Nx // 2 + 1)
        self._neg_rfft_x = (-self._rfft_x) % self.Nx
        self._wave_coefs_pos = self.wave_coefs[:, :self.Nx//2 + 1]
        self._wave_coefs_neg_conj = np.conj(self.wave_coefs[
            self._neg_y[:, np.newaxis],
            self._neg_rfft_x[np.newaxis, :]
        ])
        half_shape = (self.Ny, self.Nx//2 + 1)
        self._phase_pos = np.empty(half_shape, dtype=np.complex64)
        self._phase_neg_conj = np.empty(half_shape, dtype=np.complex64)
        self._wave_phased_pos = np.empty(half_shape, dtype=np.complex64)
        self._wave_phased_neg_conj = np.empty(half_shape, dtype=np.complex64)
        self._wave_phased = np.empty(half_shape, dtype=np.complex64)
        self._ifft_scratch = np.empty(half_shape, dtype=np.complex64)

        self._omega_pos = self.omega[:, :self.Nx//2 + 1]
        if 'V' in self.compute:
            self._iomega_pos = 1j*self._omega_pos
            self._wave_diff_t_phased = np.empty(half_shape, dtype=np.complex64)
            self._wave_diff_t_phased_neg_conj = np.empty(
                half_shape, dtype=np.complex64
            )
        if 'A' in self.compute:
            self._neg_omega2_pos = -(self._omega_pos**2)
            self._wave_diff2_t_phased = np.empty(
                half_shape, dtype=np.complex64
            )
            self._wave_diff2_t_phased_neg_conj = np.empty(
                half_shape, dtype=np.complex64
            )
        if 'Diff2' in self.compute or (
                'Diff' in self.compute and self.choppy_enable):
            self._neg_kx2 = -(self.kx*self.kx)
            self._neg_ky2 = -(self.ky*self.ky)
        if 'Diff2' in self.compute:
            self._neg_kxky = -(self.kx*self.ky)
        if 'hMTF' in self.compute:
            k_max = np.float32(2.*np.pi/20.)
            mu = np.float32(0.04/16.)
            dlnS_dlnk = np.float32(-3.)
            Vg_c = np.float32(0.5)
            M_h = (1j*self.k*self.omega*(self.omega + 1j*mu)
                   / (self.omega**2 + mu**2)
                   * (dlnS_dlnk - Vg_c)).astype(np.complex64)
            M_h[self.k > k_max] = 0.
            self._hmtf_pos = M_h[:, :self.Nx//2 + 1].copy()
            self._hmtf_pos_conj = np.conj(self._hmtf_pos)
            self._wave_hmtf_phased = np.empty(
                half_shape, dtype=np.complex64
            )

        if 'D' in self.compute:
            self.Dx = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Dy = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Dz = np.empty([self.Ny, self.Nx], dtype=np.float32)
        if 'Diff' in self.compute:
            self.Diffx = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Diffy = np.empty([self.Ny, self.Nx], dtype=np.float32)
        if 'Diff2' in self.compute:
            self.Diffxx = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Diffyy = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Diffxy = np.empty([self.Ny, self.Nx], dtype=np.float32)
        if 'V' in self.compute:
            self.Vx = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Vy = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Vz = np.empty([self.Ny, self.Nx], dtype=np.float32)
        if 'A' in self.compute:
            self.Ax = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Ay = np.empty([self.Ny, self.Nx], dtype=np.float32)
            self.Az = np.empty([self.Ny, self.Nx], dtype=np.float32)
        if 'hMTF' in self.compute:
            self.hMTF = np.empty([self.Ny, self.Nx], dtype=np.float32)

    def _ifft2_real(self, hermitian_half, multiplier=1.,
                    spectrum_neg_conj=None):
        """Transform a packed Hermitian spectrum into a real field."""
        if not np.isscalar(multiplier):
            multiplier_full = multiplier
            multiplier = multiplier_full[:, :self.Nx//2 + 1]
            np.multiply(
                multiplier, hermitian_half, out=self._ifft_scratch
            )

            # fftfreq represents an even-sized Nyquist bin with its negative
            # frequency. Correct those self-paired rows/columns explicitly so
            # odd derivative multipliers remain identical to the full ifft2.
            if spectrum_neg_conj is not None and self.Nx % 2 == 0:
                self._ifft_scratch[:, -1] += 0.5*(
                    np.conj(multiplier_full[self._neg_y, self.Nx//2])
                    - multiplier[:, -1]
                )*spectrum_neg_conj[:, -1]
            if spectrum_neg_conj is not None and self.Ny % 2 == 0:
                x_stop = -1 if self.Nx % 2 == 0 else None
                self._ifft_scratch[self.Ny//2, :x_stop] += 0.5*(
                    np.conj(multiplier_full[
                        self.Ny//2, self._neg_rfft_x[:x_stop]
                    ]) - multiplier[self.Ny//2, :x_stop]
                )*spectrum_neg_conj[self.Ny//2, :x_stop]
        else:
            np.multiply(
                multiplier, hermitian_half, out=self._ifft_scratch
            )
        return sp.fft.irfft2(
            self._ifft_scratch,
            s=(self.Ny, self.Nx),
            workers=self.workers
        )


    @property
    def t(self):
        """ Surface time property (t)
            When time (t) is updated, surface is propagated and all
            parameters are computed again.
        """

        if not self.initialized:
            raise Exception('Surface not initialized')

        return self._t

    @t.setter
    def t(self, value):

        if not self.initialized:
            raise Exception('Surface not initialized')

        self._t = np.float32(value)

        # Propagate
        np.exp(
            -1j*self._omega_pos*self._t, out=self._phase_pos
        )
        np.conjugate(self._phase_pos, out=self._phase_neg_conj)
        np.multiply(
            self._wave_coefs_pos, self._phase_pos,
            out=self._wave_phased_pos
        )
        np.multiply(
            self._wave_coefs_neg_conj, self._phase_neg_conj,
            out=self._wave_phased_neg_conj
        )
        np.add(
            self._wave_phased_pos, self._wave_phased_neg_conj,
            out=self._wave_phased
        )
        self._wave_phased *= 0.5
        if self.swell_enable:
            swell_phased = (self.swell_exp*np.exp(-1j*self.swell_omega*self._t)).astype(np.complex64)

        # HORIZ. DISPL. & HEIGHT FIELD (Dx, Dy, Dz)
        if 'D' in self.compute:
            self.Dx[:] = self._ifft2_real(self._wave_phased, 1j*self.kxn, self._wave_phased_neg_conj) + self.current[0]*self._t
            self.Dy[:] = self._ifft2_real(self._wave_phased, 1j*self.kyn, self._wave_phased_neg_conj) + self.current[1]*self._t
            self.Dz[:] = self._ifft2_real(self._wave_phased)
            if self.swell_enable:
                self.Dx += np.real(1j*self.swell_kx/self.swell_k*swell_phased).astype(np.float32)
                self.Dy += np.real(1j*self.swell_ky/self.swell_k*swell_phased).astype(np.float32)
                self.Dz += np.real(swell_phased).astype(np.float32)

        # FIRST SPATIAL DERIVATIVES - SLOPES (Diffx, Diffy)
        if 'Diff' in self.compute:
            if not self.choppy_enable:
                self.Diffx[:] = self._ifft2_real(self._wave_phased, 1j*self.kx, self._wave_phased_neg_conj)
                self.Diffy[:] = self._ifft2_real(self._wave_phased, 1j*self.ky, self._wave_phased_neg_conj)
                if self.swell_enable:
                    self.Diffx += np.real(1j*self.swell_kx*swell_phased)
                    self.Diffy += np.real(1j*self.swell_ky*swell_phased)
            else:
                aux_x = self._ifft2_real(
                    self._wave_phased, 1j*self.kx,
                    self._wave_phased_neg_conj
                )
                aux_y = self._ifft2_real(
                    self._wave_phased, 1j*self.ky,
                    self._wave_phased_neg_conj
                )
                aux_xx = self._ifft2_real(
                    self._wave_phased, self._neg_kx2*self.kinv
                )
                aux_yy = self._ifft2_real(
                    self._wave_phased, self._neg_ky2*self.kinv
                )
                self.Diffx[:] = aux_x/(1.+aux_xx)
                self.Diffy[:] = aux_y/(1.+aux_yy)
                if self.swell_enable:
                    self.Diffx += np.real(1j*self.swell_kx*swell_phased)
                    self.Diffy += np.real(1j*self.swell_ky*swell_phased)

        # SECOND SPATIAL DERIVATIVES (Diffxx, Diffyy, Diffxy)
        if 'Diff2' in self.compute:
            if not self.choppy_enable:
                self.Diffxx[:] = self._ifft2_real(self._wave_phased, self._neg_kx2)
                self.Diffyy[:] = self._ifft2_real(self._wave_phased, self._neg_ky2)
                self.Diffxy[:] = self._ifft2_real(self._wave_phased, self._neg_kxky, self._wave_phased_neg_conj)
                if self.swell_enable:
                    self.Diffxx += np.real(-self.swell_kx**2.*swell_phased)
                    self.Diffyy += np.real(-self.swell_ky**2.*swell_phased)
                    self.Diffxy += np.real(-self.swell_kx*self.swell_ky*swell_phased)
            else:
                aux_xxx = self._ifft2_real(self._wave_phased, 1j*self._neg_kx2*self.kx*self.kinv, self._wave_phased_neg_conj)
                aux_yyy = self._ifft2_real(self._wave_phased, 1j*self._neg_ky2*self.ky*self.kinv, self._wave_phased_neg_conj)
                aux_xxy = self._ifft2_real(self._wave_phased, 1j*self._neg_kx2*self.ky*self.kinv, self._wave_phased_neg_conj)
                self.Diffxx[:] = ((1.+aux_xx)*self._ifft2_real(self._wave_phased, self._neg_kx2) - aux_xxx*aux_x)/((1+aux_xx)**3.)
                self.Diffyy[:] = ((1.+aux_yy)*self._ifft2_real(self._wave_phased, self._neg_ky2) - aux_yyy*aux_y)/((1+aux_yy)**3.)
                self.Diffxy[:] = ((1.+aux_xx)*self._ifft2_real(self._wave_phased, self._neg_kxky, self._wave_phased_neg_conj) - aux_xxy*aux_x)/((1+aux_xx)**2.*(1+aux_yy))
                if self.swell_enable:
                    self.Diffxx += np.real(-self.swell_kx**2.*swell_phased)
                    self.Diffyy += np.real(-self.swell_ky**2.*swell_phased)
                    self.Diffxy += np.real(-self.swell_kx*self.swell_ky*swell_phased)

        # FIRST TIME DERIVATIVES - VELOCITY (Vx, Vy, Vz)
        if 'V' in self.compute:
            np.subtract(
                self._wave_phased_neg_conj, self._wave_phased_pos,
                out=self._wave_diff_t_phased
            )
            self._wave_diff_t_phased *= 0.5*self._iomega_pos
            np.multiply(
                self._iomega_pos, self._wave_phased_neg_conj,
                out=self._wave_diff_t_phased_neg_conj
            )
            self.Vx[:] = self._ifft2_real(self._wave_diff_t_phased, 1j*self.kxn, self._wave_diff_t_phased_neg_conj) + self.current[0]
            self.Vy[:] = self._ifft2_real(self._wave_diff_t_phased, 1j*self.kyn, self._wave_diff_t_phased_neg_conj) + self.current[1]
            self.Vz[:] = self._ifft2_real(self._wave_diff_t_phased)
            if self.swell_enable:
                swell_diff_t_phased = -1j*self.swell_omega*swell_phased
                self.Vx += np.real(1j*self.swell_kx/self.swell_k*swell_diff_t_phased)
                self.Vy += np.real(1j*self.swell_ky/self.swell_k*swell_diff_t_phased)
                self.Vz += np.real(swell_diff_t_phased)

        # SECOND TIME DERIVATIVES - ACCELERATION (Ax, Ay, Az)
        if 'A' in self.compute:
            np.multiply(
                self._neg_omega2_pos, self._wave_phased,
                out=self._wave_diff2_t_phased
            )
            np.multiply(
                self._neg_omega2_pos, self._wave_phased_neg_conj,
                out=self._wave_diff2_t_phased_neg_conj
            )
            self.Ax[:] = self._ifft2_real(self._wave_diff2_t_phased, 1j*self.kx*self.kinv, self._wave_diff2_t_phased_neg_conj)
            self.Ay[:] = self._ifft2_real(self._wave_diff2_t_phased, 1j*self.ky*self.kinv, self._wave_diff2_t_phased_neg_conj)
            self.Az[:] = self._ifft2_real(self._wave_diff2_t_phased)
            if self.swell_enable:
                swell_diff2_t_phased = -self.swell_omega**2.*swell_phased
                self.Ax += np.real(1j*self.swell_kx/self.swell_k*swell_diff2_t_phased)
                self.Ay += np.real(1j*self.swell_ky/self.swell_k*swell_diff2_t_phased)
                self.Az += np.real(swell_diff2_t_phased)

        # HYDRODYNAMIC MTF (Bertrand Chapron, personal communication)
        # TODO: Stefan Sauer should check this is correct.
        # TODO: What happens with swell?
        if 'hMTF' in self.compute:
            np.multiply(
                self._hmtf_pos, self._wave_phased_pos,
                out=self._wave_hmtf_phased
            )
            self._wave_hmtf_phased += (
                self._hmtf_pos_conj*self._wave_phased_neg_conj
            )
            self._wave_hmtf_phased *= 0.5
            self.hMTF[:] = self._ifft2_real(self._wave_hmtf_phased)
