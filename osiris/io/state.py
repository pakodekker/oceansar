
import time
from netCDF4 import Dataset
from .netcdf import NETCDFHandler


class OceanStateFile(NETCDFHandler):

    def __init__(self, file_name, mode, ocean_dim=None, format='NETCDF4'):

        self.__file__ = Dataset(file_name, mode, format)

        # If writing, define file
        if mode == 'w':
            # Set file attributes
            self.__file__.description = 'OSIRIS Library Ocean State File'
            self.__file__.history = 'Created ' + time.ctime(time.time())
            self.__file__.source = 'OSIRIS Library'

            # Dimensions
            if not ocean_dim:
                raise ValueError('Ocean dimensions are needed when creating new file!')

            self.__file__.createDimension('y_dim', ocean_dim[0])
            self.__file__.createDimension('x_dim', ocean_dim[1])

            # Variables
            Nx = self.__file__.createVariable('Nx', 'u4')
            Nx.units = '[]'
            Ny = self.__file__.createVariable('Ny', 'u4')
            Ny.units = '[]'
            Lx = self.__file__.createVariable('Lx', 'f4')
            Lx.units = '[m]'
            Ly = self.__file__.createVariable('Ly', 'f4')
            Ly.units = '[m]'
            dx = self.__file__.createVariable('dx', 'f4')
            dx.units = '[m]'
            dy = self.__file__.createVariable('dy', 'f4')
            dy.units = '[m]'
            wind_dir = self.__file__.createVariable('wind_dir', 'f4')
            wind_dir.units = '[deg]'
            wind_dir_eff = self.__file__.createVariable('wind_dir_eff', 'f4')
            wind_dir_eff.units = '[deg]'
            wind_fetch = self.__file__.createVariable('wind_fetch', 'f4')
            wind_fetch.units = '[m]'
            wind_U = self.__file__.createVariable('wind_U', 'f4')
            wind_U.units = '[m/s]'
            wind_U_eff = self.__file__.createVariable('wind_U_eff', 'f4')
            wind_U_eff.units = '[m/s]'
            current_mag = self.__file__.createVariable('current_mag', 'f4')
            current_mag.units = '[]'
            current_dir = self.__file__.createVariable('current_dir', 'f4')
            current_dir.units = '[deg]'
            swell_enable = self.__file__.createVariable('swell_enable', 'f4')
            swell_enable.units = '[]'
            swell_ampl = self.__file__.createVariable('swell_ampl', 'f4')
            swell_ampl.units = '[m]'
            swell_dir = self.__file__.createVariable('swell_dir', 'f4')
            swell_dir.units = '[deg]'
            swell_wl = self.__file__.createVariable('swell_wl', 'f4')
            swell_wl.units = '[m]'
            swell_ph0 = self.__file__.createVariable('swell_ph0', 'f4')
            swell_ph0.units = '[rad]'
            kx = self.__file__.createVariable('kx', 'f4', ('y_dim', 'x_dim'))
            kx.units = '[rad/m]'
            ky = self.__file__.createVariable('ky', 'f4', ('y_dim', 'x_dim'))
            ky.units = '[rad/m]'
            wave_coefs_r = self.__file__.createVariable('wave_coefs_r', 'f8', ('y_dim', 'x_dim'))
            wave_coefs_r.units = '[]'
            wave_coefs_i = self.__file__.createVariable('wave_coefs_i', 'f8', ('y_dim', 'x_dim'))
            wave_coefs_i.units = '[]'
            choppy_enable = self.__file__.createVariable('choppy_enable', 'f4')
            choppy_enable.units = '[]'