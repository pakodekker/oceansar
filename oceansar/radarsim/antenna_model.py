"""DRAMA antenna definitions shared by acquisition, focusing and DC modelling.

One Rx pattern describes one OceanSAR channel, including any internal arrays.
OceanSAR's b_ati/b_xti positions translate that complete antenna; DRAMA's
az_offset/el_offset remain local offsets relative to its configured origin.
Only normalized co-polar azimuth fields are used; elevation and absolute gain
are deliberately not part of the current simulator signal model.
"""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
from drama.performance.sar.antenna_patterns import pattern


from oceansar.ocs_io.sar_config import resolve_acquisition, selected as _selected


def channel_positions(cfg):
    """Return along-/cross-track receive positions in metres, each shape (N,).

    sar.num_ch counts output channels, independently of antenna element count.
    A scalar b_ati/b_xti is a regular spacing from zero; an array specifies
    signed positions relative to the existing OceanSAR geometric reference.
    """
    count = cfg.sar.num_ch
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError('sar.num_ch must be a positive integer')
    positions = []
    for name in ('b_ati', 'b_xti'):
        values = np.asarray(getattr(cfg.sar, name), dtype=float)
        if values.ndim == 0:
            values = np.arange(int(count))*values
        if values.shape != (int(count),) or not np.isfinite(values).all():
            raise ValueError(f'sar.{name} must be finite scalar spacing or num_ch positions')
        positions.append(values)
    return tuple(positions)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class SarAntenna:
    """One transmit pattern and one shared per-channel receive pattern.

    tx, rx are DRAMA pattern objects. configuration is a portable snapshot of
    the selected definitions (no filenames), saved as JSON in raw/SLC headers.
    Internal elements and offsets are evaluated by DRAMA, never converted to
    additional OceanSAR channels. No absolute G0 or elevation gain is applied.
    """
    tx: pattern
    rx: pattern
    configuration: dict

    @classmethod
    def from_config(cls, cfg):
        """Build DRAMA antennas from named sar.tx_antenna/rx_antenna sections.

        Sections use DRAMA's La/Le, type_a/type_e, Na/Ne, wa_tx/wa_rx,
        we_tx/we_rx, element_pattern and azimuth_spacing/elevation_spacing.
        Optional constructor fields (offsets, beamwidths, gain, tilt, squint)
        are forwarded. Named mode wa_tx_phase_N/we_tx_phase_N/we_rx_phase_N
        use degrees and one-based suffixes, as in DRAMA's performance code.
        Phase-only Tx mode weights replace wa_tx; Rx elevation mode phases
        multiply the selected amplitude weights. Steering is not supported.
        """
        resolve_acquisition(cfg)
        definitions = {}
        selected_mode = getattr(cfg.mode, 'definition', None)
        mcnf = getattr(cfg, selected_mode) if selected_mode not in (None, 'None') else None
        swath = int(getattr(cfg.mode, 'subswath', 0))
        names = {role: getattr(cfg.sar, f'{role}_antenna', f'{role}_ant') for role in ('tx', 'rx')}

        def build(name, role, chain=()):
            name = name.strip()
            if name in chain:
                raise ValueError('Cyclic antenna element_pattern reference: '+ ' -> '.join((*chain, name)))
            if not hasattr(cfg, name):
                raise ValueError(f'Missing antenna section [{name}] selected for {role}')
            section = getattr(cfg, name)
            values = {key: _plain(value) for key, value in vars(section).items() if key != 'elements'}
            definitions[name] = values
            required = ('La', 'Le', 'Na', 'Ne', 'type_a', 'type_e')
            missing = [key for key in required if not hasattr(section, key)]
            if missing:
                raise ValueError(f'[{name}] missing DRAMA antenna fields: {", ".join(missing)}')
            if not np.isfinite([section.La, section.Le]).all() or min(section.La, section.Le) <= 0:
                raise ValueError(f'[{name}] La and Le must be positive finite lengths in metres')
            for key in ('Na', 'Ne'):
                n = getattr(section, key)
                if isinstance(n, bool) or int(n) != n or n < 1:
                    raise ValueError(f'[{name}] {key} must be a positive integer')
            if getattr(section, 'steering_rate', 0) != 0:
                raise ValueError('Time-dependent antenna steering is not implemented in OceanSAR')
            children = getattr(section, 'element_pattern', None)
            if children is not None:
                if isinstance(children, str):
                    children = build(children, role, (*chain, name))
                else:
                    children = [build(child, role, (*chain, name)) for child in children]
                    if len(children) != (section.Na if section.type_a != 'element' else section.Ne):
                        raise ValueError(f'[{name}] element_pattern count must match the array element count')
            weights = {}
            for axis, count in (('a', int(section.Na)), ('e', int(section.Ne))):
                weight = np.asarray(getattr(section, f'w{axis}_{role}', 1))
                if not chain and role == 'rx' and axis == 'e' and mcnf is not None and hasattr(mcnf, 'we_rx'):
                    weight = np.asarray(_selected(mcnf.we_rx, swath, 'mode.we_rx'))
                phase_key = f'w{axis}_{role}_phase_{swath+1}'
                if not chain and mcnf is not None and hasattr(mcnf, phase_key):
                    phases = np.asarray(getattr(mcnf, phase_key), dtype=float)
                    if phases.shape != (count,) or not np.isfinite(phases).all():
                        raise ValueError(f'{phase_key} must contain {count} finite phases in degrees')
                    if role == 'tx':
                        weight = np.exp(1j*np.deg2rad(phases))
                    else:
                        from drama.performance.sar.antenna_patterns import w2ww
                        weight = w2ww(count, weight.item() if weight.ndim == 0 else weight).reshape(-1) * np.exp(1j*np.deg2rad(phases))
                if weight.ndim > 1 or (weight.ndim == 1 and weight.size != count) or not np.isfinite(weight).all() or not np.any(abs(weight) > 0):
                    raise ValueError(f'[{name}] w{axis}_{role} must be finite scalar or {count} nonzero weights')
                weights[f'w{axis}'] = weight.item() if weight.ndim == 0 else weight
            optional = {key: getattr(section, key) for key in (
                'beamwidth_a', 'beamwidth_e', 'az_offset', 'el_offset', 'tilt',
                'el0', 'G0', 'xpol2cpol') if hasattr(section, key)}
            squint = getattr(section, 'squint', getattr(cfg.sar, 'squint', 0))
            if mcnf is not None and hasattr(mcnf, 'squint'):
                squint = _selected(mcnf.squint, swath, 'mode.squint')
            return pattern(float(cfg.sar.f0), type_a=section.type_a, type_e=section.type_e,
                           La=section.La, Le=section.Le, Nel_a=int(section.Na), Nel_e=int(section.Ne),
                           spacing_a=getattr(section, 'azimuth_spacing', 1),
                           spacing_e=getattr(section, 'elevation_spacing', 1),
                           element_pattern=children, squint=squint, verbosity=0,
                           **weights, **optional)

        tx, rx = build(names['tx'], 'tx'), build(names['rx'], 'rx')
        return cls(tx, rx, dict(schema='drama-azimuth-v1', f0=float(cfg.sar.f0),
            squint=float(getattr(cfg.sar, 'squint', 0)), antennas=names, sections=definitions,
            mode=({key: _plain(getattr(cfg.mode, key)) for key in ('definition', 'subswath') if hasattr(cfg.mode, key)}
                  if mcnf is not None else {key: _plain(value) for key, value in vars(cfg.mode).items() if key != 'elements'}),
            mode_definition=None if mcnf is None else {key: _plain(value) for key, value in vars(mcnf).items() if key != 'elements'}))

    @classmethod
    def from_product(cls, cfg, dataset):
        """Load the acquired antenna snapshot, requiring the same carrier.

        cfg supplies current processing controls. Saved antenna sections,
        phase weights and mode selection supersede later antenna config edits.
        Products without a snapshot use the explicitly supplied configuration.
        """
        saved = getattr(dataset, 'antenna_configuration', None)
        if saved is None:
            return cls.from_config(cfg)
        data = json.loads(saved)
        if data['schema'] != 'drama-azimuth-v1' or not np.isclose(data['f0'], cfg.sar.f0, rtol=1e-12):
            raise ValueError('Raw antenna metadata schema/carrier disagrees with focusing configuration')
        config = SimpleNamespace(sar=SimpleNamespace(f0=data['f0'], squint=data['squint'],
                    **{f'{role}_antenna': name for role, name in data['antennas'].items()}),
                    mode=SimpleNamespace(**data['mode']), processing=SimpleNamespace())
        for name, values in data['sections'].items():
            setattr(config, name, SimpleNamespace(**values))
        definition = data['mode_definition']
        if definition is not None:
            setattr(config, config.mode.definition, SimpleNamespace(**definition))
        return cls.from_config(config)

    def field(self, direction_cosine):
        """Return normalized Tx times Rx complex64 co-polar azimuth field.

        direction_cosine is a dimensionless scalar/array in DRAMA's azimuth
        coordinates. Preserve its shape. Elevation, channel translation and
        cross-polar coupling are not applied here.
        """
        u = np.asarray(direction_cosine, dtype=float)
        return (self.tx.azimuth(u, field=True)[0]*self.rx.azimuth(u, field=True)[0]).astype(np.complex64)

    def save(self, dataset):
        """Store the portable acquired antenna definition in an open NetCDF."""
        dataset.antenna_configuration = json.dumps(self.configuration, sort_keys=True)
        dataset.antenna_pattern_scope = 'normalized complex co-polar azimuth field; one Rx model per channel'

    def stripmap_sine_span(self):
        """Return a finite main-beam acquisition span from the Tx half-power width.

        The span is 1.7 times the connected Tx half-power width, approximately
        the historical 1.5*lambda/La for a uniform aperture. It follows tapering
        and beamwidth overrides. This remains a finite main-beam simulation,
        not a guarantee of complete sidelobe/ambiguity illumination. Use
        mode.aperture_time (seconds) for an explicitly chosen stripmap duration.
        """
        u = np.linspace(-1., 1., 32769)
        p = abs(self.tx.azimuth(u, field=True)[0])**2
        peak = int(np.argmax(p))
        if not np.isfinite(p).all() or p[peak] <= 0:
            raise ValueError('Transmit antenna has no finite nonzero azimuth response')
        left, right = peak, peak
        while left > 0 and p[left] >= p[peak]/2:
            left -= 1
        while right < p.size-1 and p[right] >= p[peak]/2:
            right += 1
        if left == 0 or right == p.size-1:
            raise ValueError('Unbounded Tx main beam: set mode.aperture_time for stripmap')
        half = p[peak]/2
        lo = np.interp(half, p[left:left+2], u[left:left+2])
        hi = np.interp(half, p[right-1:right+1][::-1], u[right-1:right+1][::-1])
        return 1.7*(hi-lo)
