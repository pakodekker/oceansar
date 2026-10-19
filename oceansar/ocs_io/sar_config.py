"""Resolve one named acquisition mode without persisting duplicate scalars."""
import numpy as np


MODE_FIELDS = dict(mode='mode', inc_angle='inc_angle', pol='pol', prf='PRF',
                   rg_bw='pulse_bw', over_fs='over_fs', t_burst='burst_length',
                   squint='squint', aperture_time='aperture_time',
                   image_length='image_length')
PROCESSING_FIELDS = dict(doppler_bw='proc_bw', az_weighting='proc_tapering')


def selected(value, index, name):
    """Return a scalar or zero-based subswath entry from a one-dimensional value."""
    values = np.asarray(value)
    if values.ndim == 0:
        return values.item()
    if values.ndim != 1 or not 0 <= index < values.size:
        raise ValueError(f'{name} must be scalar or cover subswath {index}')
    return values[index].item()


def resolve_acquisition(cfg):
    """Populate runtime acquisition views from a single named mode definition.

    Parameters
    ----------
    cfg : ConfigFile or namespace
        mode.definition names the definition; mode.subswath selects its zero-
        based entry. Only these selectors belong in a named-mode [mode]
        section. Definition fields supply all acquisition and processing-band
        settings. ConfigFile calls this after loading an optional system file.

    Returns
    -------
    ConfigFile or namespace
        The same object. Runtime cfg.mode and processing aliases are refreshed
        on each call and are not saved as duplicate configuration entries.
        Ocean dimensions and grid spacing are never changed. Currently only
        scansar and stripmap are implemented; steering must be zero.
    """
    selector = getattr(cfg, 'mode', None)
    name = getattr(selector, 'definition', None)
    if name in (None, 'None'):
        return cfg
    if not hasattr(cfg, name):
        raise ValueError(f'Unknown mode definition [{name}]')
    duplicate = set(getattr(selector, 'elements', ())) - {'definition', 'subswath'}
    if duplicate:
        raise ValueError(f'Move {", ".join(sorted(duplicate))} from [mode] into [{name}]')
    definition = getattr(cfg, name)
    index = getattr(selector, 'subswath', 0)
    if isinstance(index, bool) or not np.isfinite(index) or int(index) != index or index < 0:
        raise ValueError('mode.subswath must be a nonnegative integer')
    index = int(index)
    get = lambda key: selected(getattr(definition, key), index, f'{name}.{key}')
    for key in ('mode', 'inc_angle', 'pol', 'PRF', 'pulse_bw', 'over_fs', 'proc_bw', 'proc_tapering'):
        if not hasattr(definition, key):
            raise ValueError(f'[{name}] needs {key}')
    kind = str(get('mode')).lower()
    if kind not in ('scansar', 'stripmap'):
        raise ValueError(f'Operating mode {kind!r} is not implemented (supported: scansar, stripmap)')
    if selected(getattr(definition, 'steering_rate', 0), index, f'{name}.steering_rate') != 0:
        raise ValueError('Time-dependent antenna steering is not implemented in OceanSAR')
    # DRAMA's performance routines consume this boolean. The enum is the sole
    # persisted mode-type setting; do not write a second scansar flag to disk.
    definition.scansar = kind == 'scansar'
    if kind == 'scansar' and not hasattr(definition, 'burst_length'):
        raise ValueError(f'[{name}] ScanSAR requires burst_length in seconds')
    for alias, source in MODE_FIELDS.items():
        if hasattr(definition, source):
            setattr(selector, alias, get(source))
        elif alias in ('t_burst', 'aperture_time', 'image_length') and hasattr(selector, alias):
            delattr(selector, alias)
    selector.mode = kind
    if hasattr(selector, 'image_length'):
        value = selector.image_length
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f'[{name}] image_length must be positive and finite, in metres')
    if not 0 < selector.inc_angle < 90:
        raise ValueError(f'[{name}] inc_angle must be between 0 and 90 degrees')
    if str(selector.pol).lower() not in ('hh', 'vv', 'dp'):
        raise ValueError(f'[{name}] pol must be hh, vv or DP')
    selector.pol = 'DP' if str(selector.pol).lower() == 'dp' else str(selector.pol).lower()
    if hasattr(definition, 'inc_near') or hasattr(definition, 'inc_far'):
        if not hasattr(definition, 'inc_near') or not hasattr(definition, 'inc_far'):
            raise ValueError(f'[{name}] needs both inc_near and inc_far')
        if not get('inc_near') <= selector.inc_angle <= get('inc_far'):
            raise ValueError('Mode incidence is outside the selected subswath incidence interval')
    for key in ('prf', 'rg_bw', 'over_fs') + (('t_burst',) if kind == 'scansar' else ()):
        value = getattr(selector, key)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'[{name}] {MODE_FIELDS[key]} must be positive and finite')
    from types import SimpleNamespace
    if not hasattr(cfg, 'processing'):
        cfg.processing = SimpleNamespace()
    duplicates = set(getattr(cfg.processing, 'elements', ())) & PROCESSING_FIELDS.keys()
    if duplicates:
        raise ValueError(f'Move {", ".join(sorted(duplicates))} from [processing] into [{name}]')
    for alias, source in PROCESSING_FIELDS.items():
        setattr(cfg.processing, alias, get(source))
    if not np.isfinite(cfg.processing.doppler_bw) or cfg.processing.doppler_bw <= 0:
        raise ValueError(f'[{name}] proc_bw must be positive and finite')
    if not np.isfinite(cfg.processing.az_weighting):
        raise ValueError(f'[{name}] proc_tapering must be finite')
    if hasattr(cfg, 'sar'):
        cfg.sar.squint = get('squint') if hasattr(definition, 'squint') else 0.
        if hasattr(cfg, 'orbit'):
            cfg.sar.alt = cfg.orbit.Horb
    return cfg


def set_mode_parameter(cfg, field, value):
    """Set one acquisition field for the selected subswath and refresh aliases.

    cfg is a loaded configuration. field is a runtime mode name (e.g. prf or
    inc_angle) or processing alias (doppler_bw, az_weighting). value is a scalar.
    For named modes the edit goes into the definition, never into [mode]. A
    scalar shared across subswaths becomes a per-subswath array when needed.
    Return cfg. Call cfg.save() to export the edited main/system bundle.
    """
    if field not in MODE_FIELDS and field not in PROCESSING_FIELDS:
        raise ValueError(f'Unknown acquisition field {field!r}')
    name = getattr(cfg.mode, 'definition', None)
    target = cfg.mode if field in MODE_FIELDS else cfg.processing
    if name in (None, 'None'):
        setattr(target, field, value)
        return cfg
    source = (MODE_FIELDS | PROCESSING_FIELDS)[field]
    definition = getattr(cfg, name)
    index = int(getattr(cfg.mode, 'subswath', 0))
    current = np.asarray(getattr(definition, source, value))
    count = max([1]+[np.asarray(getattr(definition, key)).size for key in ('PRF', 'inc_angle')])
    if current.ndim == 0 and count == 1:
        updated = value
    else:
        dtype = np.result_type(current.dtype, np.asarray(value).dtype)
        updated = np.full(count, current.item(), dtype=dtype) if current.ndim == 0 else current.astype(dtype, copy=True)
        updated[index] = value
    setattr(definition, source, updated)
    if hasattr(definition, 'elements') and source not in definition.elements:
        definition.elements.append(source)
    return resolve_acquisition(cfg)
