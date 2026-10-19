"""OceanSAR construction of the fixed-window ScanSAR Doppler response model."""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import fft as sp_fft

from oceansar import constants, ocs_io
from oceansar.l2_processing_core.dc_estimator import _pair_mask, _taper_weights
from oceansar.l2_processing_core.dc_response import (
    correct_dc_response, doppler_response, pool_doppler_response,
)


def _antenna_model(cfg, ghist, look, frequencies, f0, antennas=None):
    """Return acquired power and the actual fixed focusing equalizer power."""
    from oceansar.radarsim.antenna_model import SarAntenna
    from oceansar.sar_l1.sar_processor import geohistory_doppler_sine, processing_antenna_pattern
    antennas = antennas if antennas is not None else SarAntenna.from_config(cfg)
    sine = geohistory_doppler_sine(ghist, look, frequencies, f0)
    raw_field = antennas.field(sine)
    field = processing_antenna_pattern(cfg, raw_field)
    inverse = np.zeros_like(field, dtype=float)
    np.divide(1., abs(field)**2, out=inverse, where=abs(field) > 0)
    return abs(raw_field)**2, inverse


def write_scansar_dc_metadata(cfg, proc_file, *, ghist, look, frequencies,
                              beam_pattern, common_band_filter, v_eff,
                              raw_az0, raw_az_size, fs_az, v_ground, antennas=None):
    """Record the actual focusing model needed for ScanSAR DC calibration.

    Parameters
    ----------
    cfg : ConfigFile
        Radar/acquisition configuration supplying named DRAMA antennas and mode.
    proc_file : ProcFile
        Open output SLC, modified in place without changing image samples.
    ghist : GeoHistory
        Geometry used for focusing and antenna Doppler-angle conversion.
    look : float
        Reference look angle in radians, as used by the focuser.
    frequencies : 1D array
        Actual processing FFT frequencies in Hz, in FFT order.
    beam_pattern : 1D array
        Complex/amplitude antenna pattern divided out by the focuser.
    common_band_filter : 1D array
        Actual deramped-domain amplitude weighting, in FFT order.
    v_eff : float
        Hyperbolic range-model effective velocity in m/s.
    raw_az0 : float
        First input pulse along-track position in metres.
    raw_az_size : int
        Input pulse count before processing zero padding.
    fs_az : float
        Input/focused azimuth sampling frequency in Hz.
    v_ground : float
        Ground-track velocity in m/s.

    antennas : SarAntenna, optional
        Acquired antenna model loaded from the raw header by the focuser.
        None constructs the model from cfg.

    Returns
    -------
    None
        Save burst gate edges, geometry and sampled antenna/equalization/
        common-band power arrays in the SLC. This makes subsequent response
        calibration independent of later configuration changes.
    """
    if str(getattr(cfg.mode, 'mode', 'stripmap')).lower() != 'scansar':
        return
    dataset = proc_file.__file__
    f0 = float(cfg.sar.f0)
    # A deliberately retained noise-only image region may extend beyond the
    # simulated scene. Model the actual SLC coordinates, not ocean.Ly.
    image_start = float(proc_file.get('az0')) / v_ground
    image_stop = image_start + (dataset['slc_r'].shape[-2]-1)/fs_az
    image_half_span = max(abs(image_start), abs(image_stop))
    extent = max(fs_az, 2*v_eff**2*image_half_span
                 / (constants.c/f0*float(proc_file.get('sr0')))
                 + float(cfg.processing.doppler_bw)/2 + 150.)
    physical_f = np.linspace(-extent, extent, 8193)
    power, _ = _antenna_model(cfg, ghist, look, physical_f, f0, antennas)
    order = np.argsort(frequencies)
    inverse = np.zeros_like(beam_pattern, dtype=float)
    np.divide(1., abs(beam_pattern)**2, out=inverse, where=abs(beam_pattern) > 0)
    dataset.createDimension('dc_antenna_frequency', physical_f.size)
    dataset.createDimension('dc_processing_frequency', len(frequencies))
    for name, values, dimension, units in (
            ('dc_antenna_frequency', physical_f, 'dc_antenna_frequency', 'Hz'),
            ('dc_antenna_power', power, 'dc_antenna_frequency', '1'),
            ('dc_processing_frequency', np.asarray(frequencies)[order], 'dc_processing_frequency', 'Hz'),
            ('dc_equalization_power', inverse[order], 'dc_processing_frequency', '1'),
            ('dc_processing_power', abs(common_band_filter[order])**2, 'dc_processing_frequency', '1')):
        variable = dataset.createVariable(name, 'f8', (dimension,))
        variable[:] = values
        variable.units = units
    dataset.setncatts(dict(acquisition_mode='scansar', dc_response_model='hyperbolic-burst-v1',
        dc_burst_start_s=float(raw_az0/v_ground-.5/fs_az),
        dc_burst_stop_s=float(raw_az0/v_ground+(raw_az_size-.5)/fs_az),
        dc_effective_velocity_m_s=float(v_eff),
        dc_antenna_equalization=int(getattr(cfg.processing, 'antenna_equalization', True)),
        dc_response_range_dependent=int(getattr(cfg.processing, 'range_dependent_azimuth', True))))


def scansar_response_model(cfg, reader):
    """Load a saved ScanSAR response model or reconstruct it from explicit inputs.

    Parameters
    ----------
    cfg : ConfigFile or None
        Configuration required when the SLC has no saved model. In that case
        geometry and antenna/processing controls are reconstructed, and the
        ideal raw header supplies pulse timing when available.
    reader : ProcFile
        Open focused SLC with sampling, range, speed and reramp metadata.

    Returns
    -------
    dict
        Sampled power/derivative, fixed equalization and processing window,
        burst times in seconds, speeds in m/s and carrier/sampling in Hz.
        Includes a cache for repeated channel/polarization window geometry.
    """
    source = reader.__file__
    scalar = lambda name: float(np.asarray(reader.get(name)).item())
    from oceansar.ocs_io import product_metadata as metadata
    fs = metadata.azimuth_sampling(source)
    model = dict(fs=fs, f0=scalar('f0'), v_ground=scalar('v_ground'),
                 sr0=scalar('sr0'), rg_sampling=scalar('rg_sampling'),
                 az0=scalar('az0'), reramp=bool(scalar('reramp')), cache={})
    if getattr(source, 'dc_response_model', '') == 'hyperbolic-burst-v1':
        model.update(f=reader.get('dc_antenna_frequency'), power=reader.get('dc_antenna_power'),
                     processing_f=reader.get('dc_processing_frequency'),
                     inverse=reader.get('dc_equalization_power'),
                     window=reader.get('dc_processing_power'),
                     v_eff=float(source.dc_effective_velocity_m_s),
                     start=float(source.dc_burst_start_s), stop=float(source.dc_burst_stop_s),
                     geometry_valid=bool(source.dc_response_range_dependent),
                     provenance='saved SLC processing and burst model')
    else:
        if cfg is None:
            raise ValueError('ScanSAR DC correction needs saved SLC response metadata or its focusing configuration')
        from oceansar.radarsim.antenna_model import SarAntenna, resolve_acquisition
        cfg = metadata.focusing_configuration(resolve_acquisition(cfg), source)
        antennas = SarAntenna.from_product(cfg, source)
        from oceansar.sar_l1.sar_processor import (
            make_geohistory, estimate_effective_velocity, reference_point_geometry,
            azimuth_common_band_filter,
            processing_azimuth_fft_size, scansar_azimuth_grid, range_processing_grid,
        )
        incidence = np.deg2rad(scalar('inc_angle'))
        geometry = make_geohistory(cfg, incidence)
        v_eff = estimate_effective_velocity(cfg, incidence, ghist=geometry)
        _, look, _ = reference_point_geometry(cfg, incidence)
        raw_path = Path(cfg.sim.path)/cfg.sim.raw_file
        raw_count = int(np.floor(float(cfg.mode.t_burst)*float(cfg.mode.prf)))
        raw_fs = float(cfg.mode.prf)
        raw_start = -float(cfg.mode.t_burst)/2
        if raw_path.exists():
            raw = ocs_io.RawFile(str(raw_path), 'r')
            try:
                raw_count = raw.__file__['raw_data_r'].shape[-2]
                raw_fs = metadata.azimuth_sampling(raw.__file__)
                raw_start = float(np.asarray(raw.get('az0')).item())/model['v_ground']
            finally:
                raw.close()
        rg_count = getattr(source, 'range_fft_size', None)
        if rg_count is None:
            rg_count, _, _ = range_processing_grid(
                geometry, look, source['slc_r'].shape[-1],
                scalar('rg_sampling'), fs, model['f0'])
        far_range = model['sr0'] + (rg_count-1)*constants.c/(2*scalar('rg_sampling'))
        minimum_rate = 2*v_eff**2/((constants.c/model['f0'])*far_range)
        _, minimum_fft, _ = scansar_azimuth_grid(
            cfg.ocean.Ly, getattr(cfg.mode, 'image_length', None), fs,
            model['v_ground'], int(np.ceil(raw_count*fs/raw_fs)),
            doppler_rate=minimum_rate,
            full_image=getattr(cfg.processing, 'scansar_full_image', False))
        saved_size = getattr(source, 'azimuth_fft_size', None)
        n = (int(saved_size) if saved_size is not None else
             processing_azimuth_fft_size(cfg, minimum_fft,
                 sp_fft.next_fast_len(minimum_fft, real=False)))
        processing_f = sp_fft.fftfreq(n, 1/fs)
        order = np.argsort(processing_f)
        extent = max(fs, 2*v_eff**2*(cfg.ocean.Ly/2/model['v_ground'])
                     / (constants.c/model['f0']*model['sr0']) + cfg.processing.doppler_bw/2 + 150.)
        f = np.linspace(-extent, extent, 8193)
        power, _ = _antenna_model(cfg, geometry, look, f, model['f0'], antennas)
        _, inverse = _antenna_model(cfg, geometry, look, processing_f, model['f0'], antennas)
        window = azimuth_common_band_filter(processing_f, cfg.processing.doppler_bw,
                                           cfg.processing.az_weighting)
        model.update(f=f, power=power, processing_f=processing_f[order],
                     inverse=inverse[order], window=abs(window[order])**2,
                     v_eff=v_eff, start=raw_start-.5/raw_fs,
                     stop=raw_start+(raw_count-.5)/raw_fs,
                     geometry_valid=bool(getattr(cfg.processing, 'range_dependent_azimuth', True)),
                     provenance='reconstructed from focusing configuration and acquisition timing')
    model['derivative'] = np.gradient(model['power'], model['f'])
    return model


def _spectral_nodes(model, times, ranges, lag, samples, shift=0.):
    wavelength = constants.c/model['f0']
    velocity = model['v_eff']
    rate = lambda t: 2*velocity**2*t/(wavelength*np.hypot(ranges, velocity*t))
    # A target at eta has raw Doppler -d phi(t_raw-eta)/dt. Deramping
    # subtracts +d phi(eta)/dt, so the burst band is centred near zero in
    # the deramped frame, while the antenna is sampled near +K*eta.
    b = rate(times)
    low = -rate(model['stop']-times)-b
    high = -rate(model['start']-times)-b
    unaliased_support = high-low < model['fs']
    low, high = np.maximum(low, -model['fs']/2), np.minimum(high, model['fs']/2)
    usable = (high > low) & unaliased_support
    high = np.where(usable, high, low+1e-6)
    fractions = np.linspace(0., 1., samples)
    # The common-band filter may have rectangular edges only one processing
    # FFT bin wide. A uniform quadrature over the wider burst can miss either
    # edge asymmetrically, biasing the zero-shift reference by a fraction of a
    # bin; division by a small Doppler sensitivity then amplifies that error.
    # Include both knots of each zero/nonzero transition exactly. Pool their
    # normalised positions across spatial nodes to keep a rectangular array
    # while preserving each node's own burst support and sampling its edges.
    nonzero = np.asarray(model['window']) > 0
    changes = np.flatnonzero(nonzero[1:] != nonzero[:-1])
    if changes.size:
        knots = model['processing_f'][np.unique(np.r_[changes, changes+1])]
        edge_fractions = (knots[None, :]-low[:, None])/(high-low)[:, None]
        edge_fractions = edge_fractions[(edge_fractions > 0) & (edge_fractions < 1)]
        fractions = np.sort(np.r_[fractions, edge_fractions])
        fractions = fractions[np.r_[True, np.diff(fractions) > 32*np.finfo(float).eps]]
        fractions[0], fractions[-1] = 0., 1.
    u = low[:, None]+(high-low)[:, None]*fractions
    physical = u+b[:, None]
    usable &= ((physical-shift).min(axis=1) >= model['f'][0]) & ((physical-shift).max(axis=1) <= model['f'][-1])
    power = np.interp(physical-shift, model['f'], model['power'])
    derivative = -np.interp(physical-shift, model['f'], model['derivative'])
    wrapped = (physical+model['fs']/2) % model['fs']-model['fs']/2
    equalizer = np.interp(wrapped, model['processing_f'], model['inverse'], period=model['fs'])
    window = np.interp(u, model['processing_f'], model['window'], period=model['fs'])
    multiplier = equalizer*window*usable[:, None]
    frequency = u
    if model['reramp']:
        dt = lag/(2*model['fs'])
        # Exact phase increment, avoiding an instantaneous-frequency
        # approximation for the restored chirp at finite estimator lag.
        rplus, rminus = np.hypot(ranges, velocity*(times+dt)), np.hypot(ranges, velocity*(times-dt))
        increment = 4*velocity**2*times*dt/(rplus+rminus)
        frequency = u+(4*np.pi/wavelength*increment/(2*np.pi*lag/model['fs']))[:, None]
    return doppler_response(frequency, power*multiplier, derivative*multiplier,
                            fs_az=model['fs'], lag=lag), usable


def annotate_scansar_crlb(ds, model, valid_mask, *, range_resolution_m,
                          segment_ids=None, snr_db=None, frequency_samples=513,
                          spatial_nodes=(9, 3)):
    """Annotate a DC product with the nominal acquisition uncertainty floor.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-channel/polarization estimate_dc result. Its full input window
        defines the observation area, irrespective of estimator taper/lag.
    model : dict
        scansar_response_model output. Only acquisition timing, geometry and
        unequalized antenna power/derivative are used; processing/equalizer
        arrays are deliberately excluded from this reference bound.
    valid_mask : 2D bool array
        Finite, unmasked input SLC samples. Windows containing gaps are
        rejected: their covariance is not represented by a duration shortcut.
    range_resolution_m : float
        Independent ground-range cell width in metres, c/(2*B_rg*sin(inc))
        for the simulator's rectangular transmitted range spectrum.
    segment_ids : 1D array, optional
        Azimuth continuity labels. Windows crossing a boundary are rejected.
    snr_db : float or None, optional
        Peak nominal antenna signal PSD / known white receiver-noise PSD,
        in dB. None (default) or +inf selects the speckle-only limit. This is
        a theoretical model input, not an SNR inferred from equalized SLCs.
    frequency_samples : int, optional
        Quadrature points per acquired burst band, default 513.
    spatial_nodes : pair of positive int, optional
        Quadrature bins in the full azimuth/range support, default (9, 3).
        Bins integrate a locally stationary spectrum; they are not looks.

    Returns
    -------
    xarray.Dataset
        Copy with dc_crlb_std (Hz), information (Hz^-2), validity/reason flags,
        azimuth observation time, independent range looks, mean acquired
        bandwidth and azimuth modes. No estimator dimension: the bound is
        shared by CDE/SDE and already refers to physical Doppler, so never
        multiply it by dc_scale_factor. Unknown homogeneous signal power is
        eliminated AFTER summing local Fisher matrices. This is a Whittle
        approximation for many modes, not an exact finite-window CRB or an
        assertion that CDE/SDE attains it. Mask gaps, aliased/clipped bands,
        unsupported geometry and near-nulls in a noiseless model yield NaN.
    """
    from oceansar.l2_processing_core.dc_uncertainty import doppler_crlb

    if not np.isfinite(range_resolution_m) or range_resolution_m <= 0:
        raise ValueError('range_resolution_m must be positive finite')
    if isinstance(frequency_samples, bool) or frequency_samples < 65 or int(frequency_samples) != frequency_samples:
        raise ValueError('frequency_samples must be an integer of at least 65')
    nodes = np.asarray(spatial_nodes)
    if nodes.shape != (2,) or np.any(nodes < 1) or np.any(nodes != nodes.astype(int)):
        raise ValueError('spatial_nodes must contain two positive integers')
    snr = np.inf if snr_db is None else float(snr_db)
    if np.isnan(snr) or snr == -np.inf:
        raise ValueError('snr_db must be finite, +inf or None')
    # A fixed PSD level keeps changes of PRF from creating extra receiver
    # information. NESZ alone cannot supply this reference without a signal
    # power/calibration model, so do not infer it from processed brightness.
    with np.errstate(over='ignore', under='ignore'):
        noise = float(np.max(model['power']))*10.**(-snr/10)
    if not np.isfinite(noise):
        raise ValueError('snr_db implies an unrepresentable noise density')
    shape = (ds.sizes['azimuth'], ds.sizes['range'])
    names = ('std', 'information', 'bandwidth_hz', 'azimuth_modes')
    fields = {name: np.full(shape, np.nan) for name in names}
    reason = np.zeros(shape, dtype='uint8')
    window = np.asarray(ds.attrs['window_samples'], int)
    stride = np.asarray(ds.attrs['output_stride_samples'], int)
    valid_mask = np.asarray(valid_mask, bool)
    needed_shape = (np.asarray(shape)-1)*stride+window
    if valid_mask.ndim != 2 or np.any(np.asarray(valid_mask.shape) < needed_shape):
        raise ValueError('valid_mask must cover every DC input window')
    if segment_ids is not None and np.asarray(segment_ids).shape != (valid_mask.shape[0],):
        raise ValueError('segment_ids must match input azimuth rows')
    spacing = np.asarray(ds.attrs['pixel_spacing_m'])
    duration = window[0]/model['fs']
    range_looks = window[1]*spacing[1]/range_resolution_m
    ybins = np.array_split(np.arange(window[0]), min(int(nodes[0]), window[0]))
    xbins = np.array_split(np.arange(window[1]), min(int(nodes[1]), window[1]))
    rows, columns = np.meshgrid([a.mean() for a in ybins], [a.mean() for a in xbins], indexing='ij')
    durations, looks = np.meshgrid([len(a)/model['fs'] for a in ybins],
        [len(a)*spacing[1]/range_resolution_m for a in xbins], indexing='ij')
    wavelength = constants.c/model['f0']
    velocity = model['v_eff']
    fraction = np.linspace(0., 1., int(frequency_samples))
    cache = model.setdefault('crlb_cache', {})
    geometry_valid = (model['geometry_valid'] and
        np.isclose(spacing[0], model['v_ground']/model['fs'], rtol=1e-6))
    for i in range(shape[0]):
        y = i*stride[0]
        for j in range(shape[1]):
            x = j*stride[1]
            support = np.asarray(valid_mask[y:y+window[0], x:x+window[1]], bool)
            contiguous = segment_ids is None or np.all(
                np.asarray(segment_ids)[y:y+window[0]] == np.asarray(segment_ids)[y])
            if not support.all() or not contiguous:
                reason[i, j] |= 1
            if not geometry_valid:
                reason[i, j] |= 2
            if reason[i, j]:
                continue
            key = (y, x, tuple(window), tuple(stride), tuple(spacing),
                   float(range_resolution_m), snr, int(frequency_samples), tuple(nodes))
            if key not in cache:
                times = model['az0']/model['v_ground']+(y+rows.ravel())/model['fs']
                ranges = model['sr0']+(x+columns.ravel())*constants.c/(2*model['rg_sampling'])
                rate = lambda t: 2*velocity**2*t/(wavelength*np.hypot(ranges, velocity*t))
                b = rate(times)
                low, high = -rate(model['stop']-times)-b, -rate(model['start']-times)-b
                if (np.any(high <= low) or np.any(high-low >= model['fs'])
                        or np.any(low < -model['fs']/2) or np.any(high > model['fs']/2)):
                    cache[key] = (None, 4)
                else:
                    f = low[:, None]+(high-low)[:, None]*fraction
                    physical = f+b[:, None]
                    if physical.min() < model['f'][0] or physical.max() > model['f'][-1]:
                        cache[key] = (None, 2)
                    else:
                        power = np.interp(physical, model['f'], model['power'])
                        dp = -np.interp(physical, model['f'], model['derivative'])
                        # A noiseless sinc null makes the asymptotic log-PSD
                        # integral singular/grid dependent. Reject this model
                        # instead of silently flooring the PSD and claiming a
                        # spuriously precise prediction. Finite noise removes
                        # that singularity; finite-window covariance is future work.
                        if noise == 0 and np.any(power < 1e-6*np.max(model['power'])):
                            cache[key] = (None, 8)
                        else:
                            local = doppler_crlb(f, power, dp,
                                observation_time_s=durations.ravel(),
                                independent_range_looks=looks.ravel(), noise_density=noise)
                            dd = np.sum(local['fisher_dd'])
                            da = np.sum(local['fisher_dlog_power'])
                            aa = np.sum(local['fisher_log_power_log_power'])
                            # Sum independent local observations first: there
                            # is one homogeneous unknown NRCS for the window,
                            # not a different nuisance for each quadrature node.
                            coefficient = da/aa if aa > 0 else 0.
                            local_coefficients = np.divide(local['fisher_dlog_power'],
                                local['fisher_log_power_log_power'],
                                out=np.zeros_like(local['information']),
                                where=local['fisher_log_power_log_power'] > 0)
                            info = np.sum(local['information'] +
                                local['fisher_log_power_log_power']*(local_coefficients-coefficient)**2)
                            info = float(info) if info > 64*np.finfo(float).eps*dd else 0.
                            modes = float(np.sum(local['independent_modes'])/range_looks)
                            cache[key] = (dict(std=1/np.sqrt(info) if info > 0 else np.inf,
                                information=info, azimuth_modes=modes,
                                bandwidth_hz=modes/duration), 0 if info > 0 else 16)
            values, flag = cache[key]
            reason[i, j] |= flag
            if values is not None and flag == 0:
                for name in names:
                    fields[name][i, j] = values[name]
    result = ds.copy()
    dims = ('azimuth', 'range')
    for name, values in fields.items():
        result[f'dc_crlb_{name}'] = (dims, values)
    result['dc_crlb_reason'] = (dims, reason)
    result['dc_crlb_valid'] = (dims, (reason == 0).astype('uint8'))
    result['dc_crlb_observation_time'] = xr.DataArray(duration, attrs={'units': 's'})
    result['dc_crlb_range_looks'] = xr.DataArray(range_looks, attrs={'units': '1'})
    result.dc_crlb_std.attrs.update(units='Hz',
        long_name='Approximate nominal-acquisition Doppler CR lower bound standard deviation',
        parameter='physical acquisition Doppler shift about zero; common unknown signal power',
        ancillary_variables='dc_crlb_valid dc_crlb_reason dc_crlb_azimuth_modes',
        interpretation='Optimal unbiased estimator floor; not measured CDE/SDE uncertainty or ensemble SEM')
    result.dc_crlb_information.attrs['units'] = 'Hz-2'
    result.dc_crlb_bandwidth_hz.attrs['units'] = 'Hz'
    result.dc_crlb_azimuth_modes.attrs.update(units='1', long_name='Azimuth time-bandwidth product; not pulse count')
    result.dc_crlb_reason.attrs.update(flag_masks=np.array([1, 2, 4, 8, 16], dtype='uint8'),
        flag_meanings='gapped_or_discontinuous_window unsupported_geometry_or_spectrum aliased_or_clipped_band noiseless_near_null unidentifiable_shift')
    result.attrs.update(dc_crlb_model='nominal unequalized ScanSAR antenna spectrum; local Whittle approximation',
        dc_crlb_reference_shift_hz=0., dc_crlb_unknown_power='one homogeneous unknown NRCS per observation window',
        dc_crlb_support='full input window; estimator taper, common-band filter and equalization excluded',
        dc_crlb_range_resolution_m=float(range_resolution_m),
        dc_crlb_snr_db=float(snr), dc_crlb_snr_definition='peak nominal antenna signal PSD / white noise PSD',
        dc_crlb_assumptions='proper complex Gaussian speckle; many azimuth/range modes; locally stationary; fixed burst; no ocean broadening or ambiguities',
        dc_crlb_frequency_samples=int(frequency_samples), dc_crlb_spatial_nodes=nodes.astype(int),
        dc_crlb_noiseless_min_relative_power=1e-6,
        dc_crlb_provenance=model['provenance'])
    return result


def calibrate_scansar_dc(ds, model, valid_mask, *, taper=None, segment_ids=None,
                         min_sensitivity=.01, min_correlation=.05, max_shift_hz=50.,
                         frequency_samples=513, spatial_nodes=(9, 3)):
    """Apply the ADPM response correction on the actual DC averaging support.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-channel/single-polarization estimate_dc result. Raw dc is kept.
    model : dict
        Output of scansar_response_model, including the actual equalization.
    valid_mask : 2D bool array
        Input validity after file masks, nonfinite/nodata exclusion.
    taper : str, tuple, callable or array, optional
        Actual estimator lag-pair taper; None denotes rectangular weights.
    segment_ids : 1D array, optional
        Original azimuth continuity labels; pairs across boundaries are excluded.
    min_sensitivity, min_correlation : float, optional
        Minimum absolute gain and model correlation strength.
    max_shift_hz : float, optional
        Symmetric local inversion interval around zero; default 50 Hz.
        The model is checked at both endpoints to 5 percent linearity.
    frequency_samples : int, optional
        Quadrature samples per local spectral interval; default 513.
    spatial_nodes : pair of int, optional
        Maximum spatial bins (azimuth, range), default (9, 3). Each bin uses
        its exact surviving taper weight and weighted coordinate centre.

    Returns
    -------
    xarray.Dataset
        Raw estimates plus model baseline, g, 1/g, corrected Doppler shift
        and validity/reason flags. The local stationary fixed-window model
        includes burst timing, range curvature, periodic antenna equalization,
        deramp/reramp and processing power weights. Noise, intrinsic ocean
        spectral broadening and motion-induced window changes are not modeled.
    """
    if frequency_samples < 65 or frequency_samples != int(frequency_samples):
        raise ValueError('frequency_samples must be an integer of at least 65')
    nodes = np.asarray(spatial_nodes)
    if nodes.shape != (2,) or np.any(nodes < 1) or np.any(nodes != nodes.astype(int)):
        raise ValueError('spatial_nodes must contain two positive integers')
    lag = int(ds.attrs['lag'])
    window = np.asarray(ds.attrs['window_samples'], dtype=int)
    stride = np.asarray(ds.attrs['output_stride_samples'], dtype=int)
    pair_shape = tuple(window-np.array([lag, 0]))
    weights, _ = _taper_weights(taper, pair_shape)
    weights = np.ones(pair_shape) if weights is None else weights
    fields = {name: np.full(ds.dc.shape, np.nan) for name in ('reference_phase', 'sensitivity', 'correlation_strength')}
    domain = np.zeros(ds.dc.shape, dtype=bool)
    methods = tuple(str(value) for value in ds.estimator.values)
    ybins = np.array_split(np.arange(pair_shape[0]), min(int(nodes[0]), pair_shape[0]))
    xbins = np.array_split(np.arange(pair_shape[1]), min(int(nodes[1]), pair_shape[1]))
    for i, az in enumerate(ds.azimuth.values):
        y = i*stride[0]
        for j, ground_range in enumerate(ds.range.values):
            x = j*stride[1]
            segments = None if segment_ids is None else np.asarray(segment_ids)[y:y+window[0]]
            support = _pair_mask(valid_mask[y:y+window[0], x:x+window[1]], segments, lag)
            cache_key = (float(az), float(ground_range), tuple(window), tuple(stride), methods,
                         int(frequency_samples), tuple(nodes), float(max_shift_hz), ds.attrs['local_taper'])
            cached = model['cache'].get(cache_key) if support.all() else None
            if cached is None:
                w = weights*support
                centres, bin_weights = [], []
                for yy in ybins:
                    for xx in xbins:
                        cell = w[np.ix_(yy, xx)]
                        total = cell.sum()
                        if total > 0:
                            row = y+lag/2+np.dot(cell.sum(axis=1), yy)/total
                            column = x+np.dot(cell.sum(axis=0), xx)/total
                            centres.append((model['az0']/model['v_ground']+row/model['fs'],
                                            model['sr0']+column*constants.c/(2*model['rg_sampling'])))
                            bin_weights.append(total)
                if not centres:
                    continue
                times, ranges = np.asarray(centres).T
                response, usable = _spectral_nodes(model, times, ranges, lag, int(frequency_samples))
                pooled = pool_doppler_response(response, bin_weights, fs_az=model['fs'], lag=lag, methods=methods)
                is_linear = np.full(len(methods), bool(usable.all() and model['geometry_valid']))
                q = 2*np.pi*lag/model['fs']
                for shift in (-max_shift_hz, max_shift_hz):
                    perturbed, perturbed_usable = _spectral_nodes(model, times, ranges, lag, int(frequency_samples), shift)
                    actual = pool_doppler_response(perturbed, bin_weights, fs_az=model['fs'], lag=lag, methods=methods)
                    delta = np.angle(np.exp(1j*q*(actual['reference_phase']-pooled['reference_phase'])))/q
                    expected = pooled['sensitivity']*shift
                    is_linear &= perturbed_usable.all()
                    is_linear &= abs(delta-expected) <= np.maximum(.05*abs(expected), 1e-5)
                cached = pooled, is_linear
                if support.all():
                    model['cache'][cache_key] = cached
            pooled, is_linear = cached
            for name in fields:
                fields[name][:, i, j] = pooled[name]
            domain[:, i, j] = is_linear
    array = lambda values: xr.DataArray(values, dims=ds.dc.dims, coords=ds.dc.coords)
    result = correct_dc_response(ds, array(fields['reference_phase']), array(fields['sensitivity']),
        array(fields['correlation_strength']), model_valid=array(domain),
        min_sensitivity=min_sensitivity, min_correlation=min_correlation, max_shift_hz=max_shift_hz)
    result.attrs.update(dc_response_model='hyperbolic-burst-v1',
        dc_response_provenance=model['provenance'], dc_response_frequency_frame='stored SLC; deramp/reramp modeled',
        dc_response_physical_shift='antenna-spectrum Doppler shift relative to zero-shift acquisition; Hz',
        dc_response_assumptions='locally stationary antenna spectrum; fixed burst/processing/equalizer; noise and ocean spectral broadening omitted',
        dc_response_spatial_nodes=nodes.astype(int), dc_response_frequency_samples=int(frequency_samples),
        dc_response_quadrature='uniform interior grid plus exact common-band transition knots',
        dc_response_linearity_relative_tolerance=.05,
        dc_burst_start_s=float(model['start']), dc_burst_stop_s=float(model['stop']))
    return result
