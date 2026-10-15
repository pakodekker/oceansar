"""Compact ocean initial conditions stored through xarray/h5netcdf."""

import time
import warnings

import numpy as np
import xarray as xr


class OceanStateFile:
    """Read or write compact ocean initial conditions using xarray/h5netcdf.

    Parameters
    ----------
    file_name : str or path-like
        HDF5-backed ocean-state filename. Native complex wave_coefs retain
        their numeric precision; split-component legacy files are unsupported.
    mode : {'r', 'w'}
        Open for reading or build an in-memory dataset for writing on close.
    ocean_dim : pair of int, optional
        Grid dimensions (Ny, Nx), required in write mode. Variable shapes are
        established by set calls.

    Attributes
    ----------
    dataset : xarray.Dataset
        Open or in-memory state dataset. Contains Fourier initial conditions
        and scalar metadata, not time-dependent surface rasters.

    Notes
    -----
    Native complex storage is an HDF5 extension to the NetCDF data model.
    Use the h5netcdf backend to read these files and close explicitly.
    """

    def __init__(self, file_name, mode, ocean_dim=None):
        if mode not in ('r', 'w'):
            raise ValueError("Ocean states support only 'r' and 'w' modes")
        self.file_name = file_name
        self.mode = mode
        if mode == 'r':
            self.dataset = xr.open_dataset(file_name, engine='h5netcdf')
            if self.dataset.attrs.get('ocean_state_format') != 'complex-v1':
                self.dataset.close()
                raise ValueError('Regenerate this ocean state with the current OceanSurface.save')
        else:
            if ocean_dim is None or len(ocean_dim) != 2:
                raise ValueError('Ocean dimensions (Ny, Nx) are required')
            self.dataset = xr.Dataset(attrs={
                'ocean_state_format': 'complex-v1',
                'description': 'OceanSAR ocean initial conditions',
                'history': 'Created ' + time.ctime(),
                'source': 'OceanSAR Simulator',
            })

    def get(self, name):
        """Materialize one stored variable as a NumPy array.

        Parameters
        ----------
        name : str
            Dataset variable name; a missing variable raises KeyError.

        Returns
        -------
        numpy.ndarray
            Values in the stored dtype and shape, including zero-dimensional
            arrays for scalar variables. No unit conversion is performed.
        """
        return self.dataset[name].values

    def set(self, name, value):
        """Assign an ocean-state variable in memory before writing on close.

        Parameters
        ----------
        name : str
            Variable name. kx and ky use dimensions (x_dim,) and (y_dim,);
            wave_coefs uses (y_dim, x_dim). Other variables must be scalar.
        value : array-like or scalar
            Values with the corresponding shape; dtype is preserved. Wavenumber
            axes kx/ky are in rad/m and receive that units attribute.

        Returns
        -------
        None
            Modify dataset only; no disk write occurs until close in write mode.
        """
        dims = {'kx': ('x_dim',), 'ky': ('y_dim',),
                'wave_coefs': ('y_dim', 'x_dim')}.get(name, ())
        self.dataset[name] = xr.DataArray(np.asarray(value), dims=dims)
        if name in ('kx', 'ky'):
            self.dataset[name].attrs['units'] = 'rad/m'

    def close(self):
        """Write a pending state if opened for writing, then close the dataset.

        Returns
        -------
        None
            Write-mode datasets replace file_name through h5netcdf, preserving
            native complex variables. Read-mode datasets simply release resources.
            The dataset is closed even if writing fails; write errors propagate.
        """
        try:
            if self.mode == 'w':
                # Complex HDF5 values are intentional; xarray/h5netcdf reads
                # them directly, without split component arrays or quantization.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='You are writing invalid netcdf features.*',
                                            category=UserWarning)
                    self.dataset.to_netcdf(self.file_name, engine='h5netcdf',
                                           invalid_netcdf=True)
        finally:
            self.dataset.close()
