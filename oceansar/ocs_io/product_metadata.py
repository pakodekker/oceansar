"""Acquisition metadata shared by the raw, SLC and derived-product stages.

``prf`` is the acquisition pulse repetition frequency; ``az_sampling`` is
the rate of the samples in this particular product. Reconstruction may change
the latter without changing the former. Neither rate is read from a run config.
"""
import copy
import json

import numpy as np


def scalar(dataset, name, *, positive=False):
    """Read one finite header scalar, raising ValueError for missing/fill data."""
    if name not in dataset.variables:
        raise ValueError(f'Product header is missing {name}')
    value = np.ma.asarray(dataset[name][...])
    if value.size != 1 or np.ma.is_masked(value):
        raise ValueError(f'Product header {name} must be a finite scalar')
    result = float(value.item())
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f'Product header {name} must be finite' + (' and positive' if positive else ''))
    return result


def azimuth_sampling(dataset):
    """Return actual slow-time sampling in Hz, preferring explicit az_sampling.

    A raw acquisition or unresampled SLC may supply only prf. An explicitly
    stored invalid az_sampling is an error, never a reason to use nominal PRF.
    """
    name = 'az_sampling' if 'az_sampling' in dataset.variables else 'prf'
    return scalar(dataset, name, positive=True)


def _mode(dataset):
    saved = json.loads(getattr(dataset, 'antenna_configuration', '{}'))
    definition = saved.get('mode_definition')
    if definition is None:
        return saved.get('mode', {})
    index = int(saved['mode'].get('subswath', 0))
    return {key: np.asarray(value).item() if np.asarray(value).ndim == 0
            else np.asarray(value)[index].item()
            for key, value in definition.items()
            if key in ('mode', 'pol', 'burst_length', 'image_length')}


def burst_image_length(dataset, cfg=None):
    """Return intended ScanSAR image length in metres, or None if unannotated.

    Prefer the explicit header, then the saved acquisition mode definition.
    cfg supplies a length only when the input has no saved value.
    """
    value = getattr(dataset, 'burst_image_length_m', None)
    if value is None:
        value = _mode(dataset).get('image_length')
    if value is None:
        value = getattr(getattr(cfg, 'mode', None), 'image_length', None)
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value <= 0):
        raise ValueError('Burst image_length must be positive and finite, in metres')
    return float(value)


def acquisition_mode(dataset, cfg=None):
    """Read the operating mode from product attributes or its antenna snapshot."""
    return str(getattr(dataset, 'acquisition_mode', None) or
               _mode(dataset).get('mode') or
               getattr(getattr(cfg, 'mode', None), 'mode', 'stripmap')).lower()


def polarizations(dataset, count, cfg=None):
    """Return polarization labels in stored axis order, validating their count.

    New products carry JSON labels; existing antenna snapshots also identify
    polarization. cfg is used only for products lacking either annotation.
    """
    saved = getattr(dataset, 'polarizations', None)
    if saved is not None:
        labels = json.loads(saved)
    else:
        pol = _mode(dataset).get('pol') or getattr(getattr(cfg, 'mode', None), 'pol', None)
        labels = (['hh', 'vv'] if str(pol).lower() == 'dp' else
                  [str(pol).lower()] if pol is not None else
                  [f'pol{i}' for i in range(count)])
    if len(labels) != count or len(set(labels)) != count:
        raise ValueError('Product polarization labels do not match its polarization dimension')
    return labels


def write_sampling(dataset, sample_rate, pulse_prf):
    """Write actual azimuth sampling and acquisition PRF, both in Hz."""
    for name, value, description in (
            ('az_sampling', sample_rate, 'Azimuth sample rate of this product'),
            ('prf', pulse_prf, 'Acquisition pulse repetition frequency')):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        var = dataset.variables.get(name)
        if var is None:
            var = dataset.createVariable(name, 'f8')
        var[...] = value
        var.units = 'Hz'
        var.long_name = description


def write_geometry(dataset, cfg, labels, *, surface=None):
    """Annotate a product with acquired mode, orbit and simulation scene grid.

    cfg supplies orbit and scene geometry. When provided, surface supplies the
    actual loaded/generated grid, including optimized counts Nx/Ny. Store small
    JSON dictionaries, never surface arrays. Later focusing can restore this
    geometry even when its processing configuration has been edited.
    """
    def fields(section, names):
        return {key: np.asarray(getattr(section, key)).tolist()
                for key in names if hasattr(section, key) and key != 'elements'}
    geometry = dict(orbit=fields(cfg.orbit, vars(cfg.orbit)),
                    ocean=fields(cfg.ocean, ('Lx', 'Ly', 'dx', 'dy', 'Nx', 'Ny',
                                             'opt_res', 'fft_max_prime')))
    if surface is not None:
        geometry['ocean'].update(fields(surface, ('Lx', 'Ly', 'dx', 'dy', 'Nx', 'Ny')))
    dataset.processing_geometry = json.dumps(geometry, sort_keys=True)
    dataset.acquisition_mode = cfg.mode.mode
    dataset.polarizations = json.dumps(list(labels))
    if cfg.mode.mode == 'scansar' and hasattr(cfg.mode, 'image_length'):
        # Read from this stage's resolved acquisition view, not a nominal mode
        # definition that could disagree with the input product being focused.
        length = burst_image_length(None, cfg)
        if length is not None:
            dataset.burst_image_length_m = length


def focusing_configuration(cfg, dataset):
    """Return a private geometry config hydrated from input product headers.

    Processing choices remain in cfg. Carrier, altitude, timing and incidence
    come from scalar headers; saved orbit/scene dictionaries replace run
    configuration values. Do not resolve nominal mode aliases after this call.
    """
    result = copy.deepcopy(cfg)
    saved = json.loads(getattr(dataset, 'processing_geometry', '{}'))
    for section, values in saved.items():
        for name, value in values.items():
            setattr(getattr(result, section), name, value)
    result.sar.f0 = scalar(dataset, 'f0', positive=True)
    result.sar.alt = result.orbit.Horb = scalar(dataset, 'orbit_alt', positive=True)
    result.mode.prf = azimuth_sampling(dataset)
    result.mode.inc_angle = scalar(dataset, 'inc_angle')
    result.mode.mode = acquisition_mode(dataset, cfg)
    length = burst_image_length(dataset, cfg)
    if length is not None:
        result.mode.image_length = length
    result.mode.rg_bw = scalar(dataset, 'rg_bw', positive=True)
    result.mode.over_fs = scalar(dataset, 'rg_sampling', positive=True)/result.mode.rg_bw
    return result
