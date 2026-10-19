"""Apply calibration and receiver errors to an ideal SAR raw-data file.

The ideal samples remain in simulator field units. For now, this stage applies
its stored amplitude calibration before adding complex noise with power NESZ.
This deliberately keeps the conversion next to the noise model: a future
receiver model can instead refer its noise to uncalibrated raw units and defer
calibration until focusing.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from netCDF4 import Dataset
from drama import utils as drutils

from oceansar.memory import memory_limited
from oceansar import ocs_io
from oceansar.ocs_io import product_metadata as metadata
from oceansar.utils.random import get_sim_seed, random_generator


def validate_raw_paths(*paths):
    """Require distinct input, intermediate and output files.

    Parameters
    ----------
    *paths : str or path-like
        Product paths to compare after resolving relative paths and symlinks.
        Existing hard links to the same file are also detected.

    Returns
    -------
    None
        No files are created or modified.

    Raises
    ------
    ValueError
        Two paths resolve to the same filename or existing file.
    """
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Ideal raw, system raw and focused output must have different filenames')
    for i, path in enumerate(resolved):
        for other in resolved[i+1:]:
            if path.exists() and other.exists() and path.samefile(other):
                raise ValueError('Input and output files must not refer to the same file')


def _copy_metadata(source, target):
    """Copy the raw schema and ancillary values, without reading raw rasters."""
    target.setncatts({key: source.getncattr(key) for key in source.ncattrs()})
    for name, dimension in source.dimensions.items():
        target.createDimension(name, None if dimension.isunlimited() else len(dimension))
    for name, variable in source.variables.items():
        attrs = {key: variable.getncattr(key) for key in variable.ncattrs()}
        options = {}
        if '_FillValue' in attrs:
            options['fill_value'] = attrs.pop('_FillValue')
        filters = variable.filters() or {}
        if filters.get('zlib'):
            options.update(zlib=True, complevel=filters['complevel'],
                           shuffle=filters['shuffle'])
        copied = target.createVariable(name, variable.datatype, variable.dimensions, **options)
        copied.setncatts(attrs)
        if name not in ('raw_data_r', 'raw_data_i'):
            copied[...] = variable[...]


@memory_limited
def sar_system_errors(cfg_file, raw_file, output_file, *, block_azimuth=128):
    """Write amplitude-calibrated SAR raw data with additive receiver noise.

    Parameters
    ----------
    cfg_file : str or path-like
        Configuration supplying srg.nesz in dB and optional sim.seed for
        repeatable noise. Acquired polarization comes from the input header.
    raw_file : str or path-like
        Ideal RawFile marked raw_data_state='ideal', containing a positive
        scalar cal_factor and raw_data_r/raw_data_i with axes (polarization,
        channel, azimuth, range). The input is read without modification.
    output_file : str or path-like
        Destination RawFile, distinct from raw_file. An existing destination
        is overwritten; its parent directory is created if needed. Sample
        dtypes, dimensions and ancillary metadata are preserved.
    block_azimuth : int, optional
        Maximum pulses per processing block, default 128. Each block contains
        one polarization/channel and all range bins; must be positive.

    Returns
    -------
    pathlib.Path
        Path to the completed output file, marked raw_data_state='system_errors'
        and calibration_applied=1. The caller owns its later cleanup.

    Raises
    ------
    ValueError
        Input/output paths collide, the input is not a supported ideal raw
        product, or calibration, polarization, NESZ or block size is invalid.

    Notes
    -----
    Output samples are cal_factor * ideal + noise. Circular complex Gaussian
    noise has E[abs(noise)**2] = 10**(srg.nesz/10) in calibrated normalized
    power units; each real component has half that variance. NESZ=-inf
    applies calibration without noise. Independent polarization streams are
    repeatable for a fixed seed, including when block_azimuth changes.
    Only current ideal raw files are accepted; older files must be regenerated.
    """
    validate_raw_paths(raw_file, output_file)
    if isinstance(block_azimuth, bool) or int(block_azimuth) != block_azimuth or block_azimuth < 1:
        raise ValueError('block_azimuth must be a positive integer')
    block_azimuth = int(block_azimuth)
    cfg = ocs_io.ConfigFile(str(cfg_file))
    seed = get_sim_seed(cfg)
    nesz_db = float(cfg.srg.nesz)
    if np.isnan(nesz_db) or nesz_db == np.inf:
        raise ValueError('NESZ must be finite or -inf to disable receiver noise')
    sigma = drutils.db2lin(nesz_db, amplitude=True)/np.sqrt(2.)
    with Dataset(raw_file, 'r') as source:
        labels = metadata.polarizations(source, source['raw_data_r'].shape[0], cfg)
        if any(label not in ('hh', 'vv') for label in labels):
            raise ValueError('SAR polarization must be hh, vv or DP')
        if (getattr(source, 'raw_data_state', None) != 'ideal'
                or 'cal_factor' not in source.variables):
            raise ValueError('Regenerate ideal raw data with sar_raw before adding system errors')
        calibration = np.ma.asarray(source['cal_factor'][...])
        if (calibration.size != 1 or np.ma.is_masked(calibration)
                or not np.isfinite(calibration.item()) or calibration.item() <= 0):
            raise ValueError('Ideal raw data must contain a positive finite cal_factor')
        calibration = float(calibration.item())
        real, imag = source['raw_data_r'], source['raw_data_i']
        if real.shape != imag.shape or real.ndim != 4 or real.shape[0] != len(labels):
            raise ValueError('Expected SAR raw dimensions (polarization, channel, azimuth, range)')
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with Dataset(output_file, 'w', format=source.data_model) as target:
            _copy_metadata(source, target)
            target.raw_data_state = 'system_errors_in_progress'
            for pol_index, label in enumerate(labels):
                rng = random_generator(seed, 'sar_noise_' + label)
                for channel in range(real.shape[1]):
                    for start in range(0, real.shape[2], block_azimuth):
                        selection = (pol_index, channel,
                                     slice(start, min(start+block_azimuth, real.shape[2])), slice(None))
                        r, q = real[selection], imag[selection]
                        # Interleaved real/imaginary draws preserve the sample
                        # order when the processing block size changes.
                        noise = rng.normal(size=r.shape + (2,)) if sigma else None
                        target['raw_data_r'][selection] = r*calibration + (sigma*noise[..., 0] if sigma else 0.)
                        target['raw_data_i'][selection] = q*calibration + (sigma*noise[..., 1] if sigma else 0.)
            target.raw_data_state = 'system_errors'
            target.calibration_applied = 1
            target.system_errors_applied = 'amplitude calibration; additive complex Gaussian receiver noise'
            target.nesz_db = nesz_db
            target.noise_seed = str(seed) if seed is not None else 'random'
            target.ideal_raw_file = str(Path(raw_file).resolve())
            target.history = getattr(source, 'history', '') + '\nSystem errors applied ' + time.ctime()
    return Path(output_file)


def main(argv=None):
    """Run the standalone system-errors command, retaining its output raw file.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the program name; None uses sys.argv.
        --cfg-file is required. --raw-file and --output-file override the
        configured sim.path/raw_file and sim.path/sys_raw_file paths; the
        default system filename is raw_data_sys.nc. --block-azimuth sets the
        number of pulses processed per block.

    Returns
    -------
    int
        Zero after successful file creation. Argument parsing raises SystemExit
        for help or invalid arguments; processing exceptions propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--cfg-file', required=True)
    parser.add_argument('-r', '--raw-file')
    parser.add_argument('-o', '--output-file')
    parser.add_argument('--block-azimuth', type=int, default=128)
    args = parser.parse_args(argv)
    cfg = ocs_io.ConfigFile(args.cfg_file)
    sar_system_errors(args.cfg_file,
                      args.raw_file or str(Path(cfg.sim.path)/cfg.sim.raw_file),
                      args.output_file or str(Path(cfg.sim.path)/getattr(cfg.sim, 'sys_raw_file', 'raw_data_sys.nc')),
                      block_azimuth=args.block_azimuth)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
