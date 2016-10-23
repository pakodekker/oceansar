""" 
    WAVESIM Test - Maria 2014
    
    Testing program to obtain the dressed spectrum of Choppy waves,
    in comparsion with the theoretical linear one.
    
    Only 1 realization
"""

import numpy as np
from scipy import linalg
from trampa import utils
from osiris import spec
from osiris import spread
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import cm
from matplotlib.colors import LogNorm


# Step 1: Create ocean surface


# Selected parameters
# Lx: surface X dimension [m]
Lx = 512.
# Ly: surface X dimension [m]
Ly = 512.
# dx: surface X resolution [m]
dx = 0.5
# dy: surface Y resolution [m]
dy = 0.5
# spec_model: omnidirection spectrum model
spec_model = 'elfouhaily'
# spread_model: spreading function model
spread_model = 'elfouhaily'
# wind_dir: wind direction (rad)
wind_dir = np.deg2rad(45.)
# wind_fetch: wind fetch (m)
wind_fetch = 500.e3
# wind_U: wind speed (m/s)
wind_U = 11.
# current_mag: current magnitude
current_mag = 0.
# current_dir: current direction (rad)
current_dir = np.deg2rad(0.)
#fft_max_prime: maximum prime factor allowed in matrix sizes
fft_max_prime = 3
# zp_value: factor for zero-padding
zp_value = 4


# Grid dimensions - optimazed
Nx = np.int(Lx/dx)
Ny = np.int(Ly/dy)

Nx = utils.optimize_fftsize(Nx, fft_max_prime)
Ny = utils.optimize_fftsize(Ny, fft_max_prime)

dx = Lx/np.float(Nx)
dy = Ly/np.float(Ny)

# X-Y vector - linespace can be problematic, refinement with arange
x = np.linspace(-Lx/2., Lx/2., Nx)
x = (np.arange(x.size) - x.size/2)  * (x[1]-x[0])
y = np.linspace(-Ly/2., Ly/2., Ny)
y = (np.arange(y.size) - y.size/2)  * (y[1]-y[0])
x, y = np.meshgrid(x, y)

# Currents
current = current_mag * np.array([np.cos(current_dir), np.sin(current_dir)])
U_eff_vec = (wind_U * np.array([np.cos(wind_dir), np.sin(wind_dir)]) - current)
wind_U_eff = linalg.norm(U_eff_vec)
wind_dir_eff = np.arctan2(U_eff_vec[1], U_eff_vec[0])

# Kx-Ky meshgrid 
kx = 2.*np.pi*np.fft.fftfreq(Nx, dx)
ky = 2.*np.pi*np.fft.fftfreq(Ny, dy)
kx, ky = np.meshgrid(kx, ky)

# Kx-Ky resolution
kx_res = kx[0, 1] - kx[0, 0]
ky_res = ky[1, 0] - ky[0, 0]

# K-theta meshgrid (Polar, wind direction shifted)
k = np.sqrt(kx**2 + ky**2)
good_k = np.where(k > np.min(np.array([kx_res, ky_res]))/2.0)
kinv = np.zeros(k.shape)
kinv[good_k] = 1./k[good_k]
theta = np.angle(np.exp(1j * (np.arctan2(ky, kx) - wind_dir_eff)))

# Compute directional wave spectrum (1/k*S(k)*D(k,theta))
wave_spec = np.zeros(k.shape)
wave_spec[good_k] = spec.models[spec_model](k[good_k], wind_U_eff, wind_fetch)
wave_spread = np.zeros(k.shape)
wave_spread[good_k] = spread.models[spread_model](k[good_k], theta[good_k],
                                                  wind_U_eff, wind_fetch)
wave_dirspec = kinv*wave_spec*wave_spread
    
# Spectrum with zero padding
wave_dirspec = np.fft.fftshift(wave_dirspec)

wave_dirspec_zp = np.zeros([zp_value*Ny,zp_value*Nx])
wave_dirspec_zp[0:Ny,0:Nx] = wave_dirspec
    
wave_dirspec = np.roll(np.roll(wave_dirspec, -Nx/2, axis=1), -Ny/2, axis=0)
wave_dirspec_zp = np.roll(np.roll(wave_dirspec_zp, -Nx/2, axis=1), -Ny/2, axis=0)
    
# new x-y 
x_new = np.linspace(-Lx/2., Lx/2., zp_value*Nx)
y_new = np.linspace(-Ly/2., Ly/2., zp_value*Ny)
x_new, y_new = np.meshgrid(x_new, y_new)
    
# new Kx-Ky
kx_new = 2.*np.pi*np.fft.fftfreq(zp_value*Nx, dx/zp_value)
ky_new = 2.*np.pi*np.fft.fftfreq(zp_value*Ny, dy/zp_value)
kx_new, ky_new = np.meshgrid(kx_new, ky_new)
    
# new x-y resolution
x_res = x_new[0, 1] - x_new[0, 0]
y_res = y_new[1, 0] - y_new[0, 0]
    
# new Kx-Ky resolution - same than before!
kx_res_new = kx_new[0, 1] - kx_new[0, 0]
ky_res_new = ky_new[1, 0] - ky_new[0, 0]

# new K-theta meshgrid (Polar, wind direction shifted)
k_new = np.sqrt(kx_new**2 + ky_new**2)
good_k_new = np.where(k_new > np.min(np.array([kx_res_new, ky_res_new]))/2.0)
kinv_new = np.zeros(k_new.shape)
kinv_new[good_k_new] = 1./k_new[good_k_new]
theta_new = np.angle(np.exp(1j * (np.arctan2(ky_new, kx_new) - wind_dir_eff)))
    
# Complex Gaussian to randomize spectrum coefficients
random_cg = 1./np.sqrt(2.)*(np.random.normal(0., 1., size=[zp_value*Ny, zp_value*Nx]) +
                            1j*np.random.normal(0., 1., size=[zp_value*Ny, zp_value*Nx]))

# Initialize coefficients
wave_coefs = zp_value**2*Nx*Ny*np.sqrt(2.*wave_dirspec_zp*kx_res_new*ky_res_new)*random_cg

# HORIZ. DISPL. & HEIGHT FIELD (Dx, Dy, Dz)
Dx = np.real(np.fft.ifft2(1j*kx_new*kinv_new*wave_coefs))
Dy = np.real(np.fft.ifft2(1j*ky_new*kinv_new*wave_coefs))
Dz = np.real(np.fft.ifft2(wave_coefs))
    
print 'The mean and variance of original Dz are: ', np.mean(Dz), np.var(Dz)
                                                        
#spectrum_Dz = utils.smooth((np.abs(np.fft.fftshift(np.fft.fft2(Dz))))**2., window_len=7)
#spectrum_Dz = np.roll(np.roll(spectrum_Dz, -(zp_value/2)*Nx, axis=1), -(zp_value/2)*Ny, axis=0)
#spectrum_Dz = (2.*spectrum_Dz) / ((zp_value**2*Nx*Ny)**2*(kx_res_new*ky_res_new))
#
#plt.figure()
#plt.imshow(np.fft.fftshift(spectrum_Dz), origin='lower', cmap=cm.jet,
#           norm=LogNorm(vmin=1.e-5, vmax=1.e2))
#plt.colorbar()
#plt.title('[Original] Dressed spectrum')
#plt.show()


                                                       
# Step 2: Obtain the real choppy surface

# Irregular x-y grid
x_irr, y_irr = (x_new + Dx, y_new + Dy)

# Interpolate using Delaunay triangularizations to the regular grid
z = mlab.griddata(x_irr.flatten(), y_irr.flatten(), Dz.flatten(), x, y, interp='linear')

# Remove possible 'masked' values
z = np.where(z.mask == True, 0.0, z)

print 'The mean and variance of interpolated Dz are: ', np.mean(z), np.var(z)

plt.figure()
plt.hist(Dz.flatten(), bins=1000, normed=True, histtype='step', align='mid', color='blue', label='Original')
plt.hist(z.flatten(), bins=1000, normed=True, histtype='step', align='mid', color='red', label='Interpolated')
v = [-1.5, 1.5, 0.0, 0.9]
plt.axis(v)
plt.title('Histogram Dz')
plt.xlabel('Dz (m)')
plt.ylabel('Pdf')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# Create a 2-D Hanning window    
win_x = np.hanning(Nx)
win_y = np.hanning(Ny)
win_2D = np.sqrt(np.outer(win_y,win_x))
  
# Insert the window to the surface    
z_new = z * win_2D

# Plot the results
plt.figure()
plt.imshow(z_new, origin='lower', cmap=utils.sea_cmap)
plt.colorbar()
plt.title('Interpolated-regular height field')
plt.show()
    
# New wave directional spectrum (wave_spectrum=smooth(abs(fft(Dz))^2))
# Notice that factor 2 is for the amplitude correction due to Hanning window
wave_spectrum = utils.smooth((np.abs(np.fft.fftshift(2.*np.fft.fft2(z_new))))**2., window_len=3)
wave_spectrum = np.roll(np.roll(wave_spectrum, -Nx/2, axis=0), -Ny/2, axis=1)
                                 
# Normalization of the spectrum (opposite to the theoretical case)
wave_spectrum = (2.*wave_spectrum)/((Nx*Ny)**2*(kx_res*ky_res))

# Only waves travelling in one main direction: half spectrum - factor of 2
wave_dirspec2 = np.zeros(k.shape)
wave_dirspec2[good_k] = wave_spectrum[good_k]
wave_dirspec2 = np.where((theta > -np.pi/2.) & (theta < np.pi/2.),
                         wave_dirspec2, 0)


# Step 3: Plots for the comparison of dressed and undressed spectrums

# Plots
plt.figure()
plt.imshow(np.fft.fftshift(wave_dirspec), origin='lower', cmap=cm.jet,
           norm=LogNorm(vmin=1.e-5, vmax=1.e2))
plt.colorbar()
plt.title('[Original] Undressed spectrum')
plt.show()
    
#plt.figure()
#plt.imshow(np.fft.fftshift(wave_dirspec_zp), origin='lower', cmap=cm.jet,
#           norm=LogNorm(vmin=1.e-5, vmax=1.e2))
#plt.colorbar()
#plt.title('[0-Padding] Undressed spectrum')
#plt.show()
    
plt.figure()
plt.imshow(np.fft.fftshift(wave_dirspec2), origin='lower', cmap=cm.jet,
           norm=LogNorm(vmin=1.e-5, vmax=1.e2))
plt.colorbar()
plt.title('[0-Padding] Dressed spectrum')
plt.show()

plt.figure()
plt.loglog(np.diag(k[0:Ny/2,0:Nx/2]), np.diag(wave_dirspec2[0:Ny/2,0:Nx/2]), 
           color='blue', label='Dressed')
plt.loglog(np.diag(k[0:Ny/2,0:Nx/2]), np.diag(wave_dirspec[0:Ny/2,0:Nx/2]), 
           color='red', label='Undressed')
v = [1.e-2, 1.e2, 1.e-8, 1.e2]
plt.axis(v)
plt.title('Diagonal with '+r'$\theta=\pi/4$'+' (rad)')
plt.xlabel('Wave number '+r'$k$' + ' (rad/m)')
plt.ylabel('Wave directional spectrum '+r'$\Phi(k)$')
plt.grid(True)
plt.legend(loc='best')
plt.show()

plt.figure()
plt.loglog(k[1,0:Nx/2], wave_dirspec2[1,0:Nx/2], color='blue', label='Dressed')
plt.loglog(k[1,0:Nx/2], wave_dirspec[1,0:Nx/2], color='red', label='Undressed')
v = [1.e-2, 1.e2, 1.e-8, 1.e2]
plt.axis(v)
plt.title('Horizontal with '+r'$\theta=\pi/4$'+' (rad)')
plt.xlabel('Wave number '+r'$k$' + ' (rad/m)')
plt.ylabel('Wave directional spectrum '+r'$\Phi(k)$')
plt.grid(True)
plt.legend(loc='best')
plt.show()

plt.figure()
plt.loglog(k[0:Ny/2,1], wave_dirspec2[0:Ny/2,1], color='blue', label='Dressed')
plt.loglog(k[0:Ny/2,1], wave_dirspec[0:Ny/2,1], color='red', label='Undressed')
v = [1.e-2, 1.e2, 1.e-8, 1.e2]
plt.axis(v)
plt.title('Vertical with '+r'$\theta=\pi/4$'+' (rad)')
plt.xlabel('Wave number '+r'$k$' + ' (rad/m)')
plt.ylabel('Wave directional spectrum '+r'$\Phi(k)$')
plt.grid(True)
plt.legend(loc='best')
plt.show()
