"""NetCDF encoding helpers for complex cross-spectra."""

import numpy as np


def quantize_significant(values, significant_digits=3):
    """Round finite values to a requested number of significant digits."""
    values = np.asarray(values)
    quantized = np.zeros_like(values)
    finite_nonzero = np.isfinite(values) & (values != 0)
    if np.any(finite_nonzero):
        magnitude = np.abs(values[finite_nonzero])
        exponent = np.floor(np.log10(magnitude))
        factor = 10. ** (significant_digits - 1 - exponent)
        quantized[finite_nonzero] = (
            np.round(values[finite_nonzero] * factor) / factor)
    quantized[~np.isfinite(values)] = np.nan
    return quantized


def split_complex_spectra(dataset, variable='wxspec',
                          significant_digits=3):
    """Replace a complex spectrum with quantized real and imaginary parts."""
    parts = dataset.copy()
    spectrum = parts[variable]
    parts[f'{variable}_re'] = (
        spectrum.dims,
        quantize_significant(
            spectrum.real.values, significant_digits).astype(np.float32))
    parts[f'{variable}_im'] = (
        spectrum.dims,
        quantize_significant(
            spectrum.imag.values, significant_digits).astype(np.float32))
    return parts.drop_vars(variable)


def packed_int16_encoding(dataset, variables, compression_level=6):
    """Create S1D-style packed-int16 and gzip NetCDF encodings."""
    encoding = {}
    for variable in variables:
        values = dataset[variable].values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            data_min = 0.
            data_max = 1.
        else:
            data_min = float(finite.min())
            data_max = float(finite.max())
            if data_max == data_min:
                data_max = data_min + max(abs(data_min), 1.) * 1e-6
        encoding[variable] = {
            'dtype': 'int16',
            # Reserve -32768 for missing data and map finite values onto
            # the remaining signed-int16 codes (-32767 through 32767).
            'scale_factor': (data_max - data_min) / (2**16 - 2),
            'add_offset': data_min + (data_max - data_min) / 2,
            '_FillValue': -32768,
            'compression': 'gzip',
            'compression_opts': compression_level,
            'shuffle': True,
        }
    return encoding


def save_compressed_xspec(dataset, output_file, significant_digits=3):
    """Write a complex cross-spectrum dataset as compressed NetCDF."""
    parts = split_complex_spectra(
        dataset, significant_digits=significant_digits)
    variables = ['wxspec_re', 'wxspec_im']
    encoding = packed_int16_encoding(parts, variables)
    parts.to_netcdf(output_file, encoding=encoding, engine='h5netcdf')


def load_compressed_xspec(input_file):
    """Load a compressed NetCDF and reconstruct its complex spectrum."""
    import xarray as xr

    with xr.open_dataset(input_file, engine='h5netcdf') as parts:
        dataset = parts.load()
    dataset['wxspec'] = (
        dataset['wxspec_re'] + 1j * dataset['wxspec_im']
    ).astype(np.complex64)
    return dataset.drop_vars(['wxspec_re', 'wxspec_im'])
