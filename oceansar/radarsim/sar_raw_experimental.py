"""Opt-in ScanSAR raw generator for guard and interpolation experiments.

This strategy reuses the physical generator, including its known-current
reference, and supplies experimental sampling/reconstruction controls. Coarse
fields can be retained so different interpolators see identical input signals.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import json
from pathlib import Path
import shutil

import h5py
from netCDF4 import Dataset
import numpy as np
from scipy import fft, signal, interpolate
from tqdm import tqdm

from oceansar.memory import memory_limited
from oceansar import constants, ocs_io
from oceansar.radarsim.factorize_raw import factorize_raw_params, aggregate_factorized_raw
from oceansar.radarsim.sar_raw import sar_raw


@dataclass(frozen=True)
class RawTestOptions:
    """Experimental controls; counts refer to coarse samples, times to seconds.

    guards is the requested minimum count on each side before FFT padding.
    interpolation is fft (legacy periodic), sinc (finite Kaiser-sinc FIR), or
    sinc_cubic (FIR upsampling by sinc_oversampling, then cubic evaluation).
    sinc_half_width is the FIR radius in coarse sample intervals; beta is the
    Kaiser shape parameter. Finite FIR methods zero-extend beyond supplied
    samples, never wrap. sinc_backend selects identical matrix (default) or
    direct polyphase evaluation; it does not change the interpolation kernel.
    scene_epoch_s fixes the evolving scene time at burst
    centre, and the snapshot time for frozen scenes. None retains legacy scene
    timing. save_coarse explicitly opts into lossless native-complex
    factorization_inputs.h5; default False avoids multi-GiB replay files.
    """
    guards: int = 4
    interpolation: str = 'fft'
    sinc_half_width: int = 8
    beta: float = 6.
    sinc_oversampling: int = 4
    sinc_backend: str = 'matrix'
    scene_epoch_s: float | None = 1.
    save_coarse: bool = False

    def __post_init__(self):
        for name, minimum in (('guards', 0), ('sinc_half_width', 1), ('sinc_oversampling', 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.interpolation not in ('fft', 'sinc', 'sinc_cubic'):
            raise ValueError('interpolation must be fft, sinc or sinc_cubic')
        if self.sinc_backend not in ('matrix','polyphase'):
            raise ValueError('sinc_backend must be matrix or polyphase')
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if self.scene_epoch_s is not None and not np.isfinite(self.scene_epoch_s):
            raise ValueError('scene_epoch_s must be finite or None')


def _finite_residual(samples,factor,options):
    """Reference finite-sinc/polyphase and optional cubic linear operator."""
    count=samples.shape[1]
    up = factor if options.interpolation == 'sinc' else options.sinc_oversampling
    kernel = signal.firwin(2*options.sinc_half_width*up+1, 1/up,
                           window=('kaiser', options.beta))
    oversampled = signal.resample_poly(samples, up, 1, axis=1,
                                       window=kernel, padtype='constant')
    if options.interpolation == 'sinc':
        return oversampled
    spline = interpolate.CubicSpline(np.arange(count*up)/up, oversampled, axis=1)
    return spline(np.arange(count*factor)/factor)


@lru_cache(maxsize=8)
def _finite_operator(count,factor,options):
    # FIR upsampling and CubicSpline evaluation at fixed nodes are both
    # linear in the samples. Their responses to the identity basis therefore
    # give exactly the same finite-boundary operator, including removed group
    # delay and spline endpoint conditions. Reuse it across all range profiles
    # and blocks instead of repeatedly constructing splines or scalar FIR loops.
    operator=_finite_residual(np.eye(count)[None],factor,options)[0]
    operator=np.asarray(operator,np.complex128)
    operator.flags.writeable=False
    return operator


def interpolate_residual(samples, factor, options, *, workers=1):
    """Upsample complex fields on axis 1, preserving the first sample's time.

    samples has axes (channel, coarse time, range); factor is the positive
    integer full/coarse sampling ratio. Return (channel, coarse time*factor,
    range). The FFT path deliberately reproduces the production packing,
    including its even-length Nyquist convention. FIR group delay is removed
    by resample_poly; a symmetric odd kernel has radius sinc_half_width coarse
    intervals. Edges with incomplete support must be assessed separately.
    matrix applies the identical finite operator using BLAS (default); polyphase
    retains its direct SciPy evaluation. Large operators fall back to direct
    evaluation to keep the cached matrix below 16 MB per option set.
    """
    if int(factor) != factor or factor < 1:
        raise ValueError('factor must be a positive integer')
    factor = int(factor)
    if factor == 1:
        return np.array(samples, copy=True)
    count = samples.shape[1]
    if options.interpolation == 'fft':
        padded = np.zeros((samples.shape[0], count*factor, samples.shape[2]), np.complex64)
        padded[:, :count] = factor*fft.fftshift(fft.fft(samples, axis=1, workers=workers), axes=1)
        return fft.ifft(np.roll(padded, -(count//2), axis=1), axis=1, workers=workers)
    # Explicit finite support makes the guard requirement reviewable. A broad
    # spectrum needs a transition band; a tone test cannot establish this
    # filter's passband fidelity or its suitability for ocean scattering.
    if options.sinc_backend=='matrix' and count*count*factor<=1_000_000:
        operator=_finite_operator(count,factor,options)
        flat=samples.transpose(1,0,2).reshape(count,-1)
        result=(operator@flat).reshape(count*factor,samples.shape[0],samples.shape[2]).transpose(1,0,2)
        return result if np.iscomplexobj(samples) else result.real
    return _finite_residual(samples,factor,options)


def _reconstruct(hh, vv, full_reference, params, options, workers):
    """Restore range/phase block by block from arrays or HDF5 field datasets.

    hh/vv have axes (channel, coarse time, block, range), or are None.
    full_reference is the exact per-block range offset in metres, with axes
    (full time, block). params supplies sample counts, k0 (rad/m), Fs (Hz),
    and retained-pulse crop. Return the two cropped complex64 raw arrays.
    """
    source = vv if vv is not None else hh
    count = params['az_steps']*params['n_pulses_b']
    ranges = source.shape[-1]
    padded_ranges = fft.next_fast_len(ranges)
    shape = (source.shape[0], count, padded_ranges)
    results = [None if field is None else np.zeros(shape, np.complex64) for field in (hh, vv)]
    range_frequency = fft.fftfreq(padded_ranges).astype(np.float32)[None, None, :]
    for block in tqdm(range(params['nblocks']), desc='Experimental reconstruction'):
        reference = np.asarray(full_reference[:, block])
        phase = np.exp(-2j*params['k0']*reference).astype(np.complex64)[None, :, None]
        shift = (reference*2/constants.c*params['Fs'])[None, :, None]
        range_phase = np.exp(-2j*np.pi*shift*range_frequency).astype(np.complex64)
        for field, result in zip((hh, vv), results):
            if field is None:
                continue
            residual = np.asarray(field[:, :, block, :])
            fine = interpolate_residual(residual, params['n_pulses_b'], options, workers=workers)
            fine = np.asarray(fine, np.complex64)
            fine *= phase
            transformed = fft.fft(fine, n=padded_ranges, axis=2, workers=workers)
            transformed *= range_phase
            result += fft.ifft(transformed, axis=2, workers=workers)
    start = params['az_crop_start']
    stop = start+params['output_az_steps']
    return tuple(None if result is None else result[:, start:stop, :ranges] for result in results)


class _Strategy:
    def __init__(self, options, inputs):
        self.options, self.inputs, self.audit = options, inputs, {}

    def parameters(self, cfg, params, surface, info):
        params = factorize_raw_params(cfg, params, surface, info,
            scansar_guard_coarse_samples=self.options.guards)
        if self.options.scene_epoch_s is not None:
            params['scene_time_offset'] = self.options.scene_epoch_s+params['az0']/params['v_ground']
            params['frozen_scene_time'] = self.options.scene_epoch_s
        full_count = params['az_steps']*params.get('n_pulses_b', 1)
        retained_centre = params['output_az0']/params['v_ground']+(params['output_az_steps']-1)/(2*cfg.mode.prf)
        self.audit = dict(options=asdict(self.options), coarse_count=params['az_steps'],
            full_count=full_count, output_count=params['output_az_steps'],
            factor=params.get('n_pulses_b', 1),
            coarse_centre_relative_to_burst_s=params['az0']/params['v_ground']
                +(params['az_steps']-1)*params['t_step']/2-retained_centre,
            retained_burst_centre_s=retained_centre,
            scene_time_offset_s=params.get('scene_time_offset', 0.),
            current_reference=dict(
                enabled=bool(cfg.srg.factorize and getattr(cfg.srg, 'factorize_current', True)),
                current_vector_m_s=(np.asarray(surface.current).tolist()
                    if hasattr(surface, 'current') else None),
                frozen=bool(getattr(cfg.ocean, 'frozen_ocean', False)) if hasattr(cfg, 'ocean') else False,
                definition='scene_time*(Ux*sin_incidence_near + Uy*sin_azimuth_block)'))
        return params

    def aggregate(self, hh, vv, coarse_reference, full_reference, params, surface, cfg, info, workers=4):
        self.audit['coarse_profile_dtype'] = str((vv if vv is not None else hh).dtype)
        if self.options.save_coarse:
            keys = ('nblocks', 'az_steps', 'n_pulses_b', 'k0', 'Fs', 'az_crop_start',
                    'output_az_steps', 't_step', 'az0', 'output_az0', 'v_ground')
            encoded = {key: np.asarray(params[key]).item() for key in keys}
            with h5py.File(self.inputs, 'x') as output:
                output.attrs['format'] = 'oceansar-factorization-test-v1'
                output.attrs['parameters_json'] = json.dumps(encoded)
                output.attrs['audit_json'] = json.dumps(self.audit)
                output.attrs['config_path'] = str(Path(cfg.cfg_file_name).resolve())
                for name, field in (('hh', hh), ('vv', vv)):
                    if field is not None:
                        output.create_dataset(name, data=field)
                output.create_dataset('coarse_reference_m', data=coarse_reference)
                output.create_dataset('full_reference_m', data=full_reference)
        if self.options.interpolation == 'fft':
            return aggregate_factorized_raw(hh, vv, coarse_reference, full_reference,
                                              params, surface, cfg, info, workers=workers)
        return _reconstruct(hh, vv, full_reference, params, self.options, workers)


@memory_limited
def sar_raw_test(cfg_file, output_file, ocean_file, reuse_ocean_file=False, *,
                 options=None, plot_save=True):
    """Generate an experimental raw product and return its timing-grid audit.

    Paths and physical configuration follow sar_raw. options is RawTestOptions
    (default four guards, legacy FFT, aligned scene epoch 1 s). Output paths
    must be fresh. An optional lossless factorization_inputs.h5 is saved beside
    the raw file, enabling replay without ocean generation. The raw product is
    ideal/unscaled; system errors and focusing remain explicit separate steps.
    This function does not modify cfg_file or an existing ocean realization.
    """
    options = RawTestOptions() if options is None else options
    output_file = Path(output_file)
    inputs = output_file.parent/'factorization_inputs.h5'
    if output_file.exists() or (options.save_coarse and inputs.exists()):
        raise FileExistsError('Use fresh experimental output paths')
    cfg = ocs_io.ConfigFile(str(cfg_file))
    if str(cfg.mode.mode).lower() != 'scansar':
        raise ValueError('Experimental generator currently requires ScanSAR')
    strategy = _Strategy(options, inputs)
    sar_raw(str(cfg_file), str(output_file), str(ocean_file), reuse_ocean_file,
            plot_save=plot_save, factorization=strategy)
    with Dataset(output_file, 'a') as raw:
        raw.raw_generator_variant = 'experimental-factorization-v1'
        raw.factorization_test_json = json.dumps(strategy.audit)
    (output_file.parent/'raw_test_audit.json').write_text(json.dumps(strategy.audit, indent=2)+'\n')
    return strategy.audit


@memory_limited
def reinterpolate(inputs_file, template_raw, output_raw, *, options, workers=4, cfg_file=None):
    """Replay saved coarse fields with a different interpolation scheme.

    inputs_file is factorization_inputs.h5; template_raw is its matching raw
    file, supplying all non-sample metadata. output_raw must be new. options
    selects the interpolation only: guards and scene epoch remain those in the
    saved audit. Return output_raw as a Path. Reconstruct both polarizations,
    apply the generator's final retained-range crop, copy template metadata,
    and replace its complex samples without changing
    the original files. The ideal output still needs system-error processing.
    cfg_file optionally supplies the memory budget. When omitted, recover the
    original configuration path saved in inputs_file when it still exists.
    """
    output_raw = Path(output_raw)
    if output_raw.exists():
        raise FileExistsError(output_raw)
    if cfg_file is None:
        with h5py.File(inputs_file,'r') as saved:
            inferred = saved.attrs.get('config_path')
        if inferred is not None and Path(inferred).is_file():
            return reinterpolate(inputs_file,template_raw,output_raw,options=options,
                                  workers=workers,cfg_file=inferred)
    with h5py.File(inputs_file, 'r') as saved:
        if saved.attrs['format'] != 'oceansar-factorization-test-v1':
            raise ValueError('Unsupported factorization input format')
        params = json.loads(saved.attrs['parameters_json'])
        audit = json.loads(saved.attrs['audit_json'])
        source_fields=[saved[name] for name in ('hh','vv') if name in saved]
        with Dataset(template_raw) as original:
            output_shape=original['raw_data_r'].shape
            expected=(len(source_fields),source_fields[0].shape[0],params['output_az_steps'])
            if output_shape[:-1]!=expected or not 0 < output_shape[-1] <= source_fields[0].shape[-1]:
                raise ValueError('Template shape differs from reconstructed raw samples')
            if json.loads(original.factorization_test_json) != audit:
                raise ValueError('Template acquisition audit differs from saved coarse fields')
        hh, vv = _reconstruct(saved.get('hh'), saved.get('vv'), saved['full_reference_m'],
                              params, options, workers)
    # The physical generator allocates FFT-padded range profiles, restores
    # phase/RCM on that grid, then writes only :rg_samp_orig. Saved coarse
    # fields retain the padded width; the matching raw template specifies the
    # final prefix crop. Cropping before reconstruction would change the range
    # FFT/shift and therefore the signal, so repeat this final crop only here.
    fields = np.stack([field[..., :output_shape[-1]] for field in (hh, vv) if field is not None])
    shutil.copy2(template_raw, output_raw)
    audit['replay_options'] = asdict(options)
    with Dataset(output_raw, 'a') as target:
        target['raw_data_r'][:] = fields.real
        target['raw_data_i'][:] = fields.imag
        target.factorization_test_json = json.dumps(audit)
        target.factorization_inputs = str(Path(inputs_file).resolve())
    return output_raw


def main():
    """Generate experimental raw data, or replay saved coarse fields from CLI."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='Generate a fresh experimental ideal raw product')
    generate.add_argument('--config', type=Path, required=True)
    generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--ocean', type=Path, required=True)
    generate.add_argument('--reuse-ocean', action='store_true')
    generate.add_argument('--guards', type=int, default=4)
    generate.add_argument('--scene-epoch-s', type=float, default=1.)
    generate.add_argument('--legacy-scene-clock', action='store_true')
    generate.add_argument('--save-coarse', action='store_true',
        help='Keep multi-GiB coarse fields for interpolation replay; default is not to save')
    replay = commands.add_parser('replay', help='Reconstruct a new ideal raw from identical coarse inputs')
    replay.add_argument('--inputs', type=Path, required=True)
    replay.add_argument('--template', type=Path, required=True)
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--config', type=Path, help='Memory controls; defaults to the saved acquisition configuration')
    replay.add_argument('--workers', type=int, default=4)
    for command in (generate, replay):
        command.add_argument('--interpolation', choices=('fft','sinc','sinc_cubic'), default='fft')
        command.add_argument('--sinc-half-width', type=int, default=8)
        command.add_argument('--beta', type=float, default=6.)
        command.add_argument('--sinc-oversampling', type=int, default=4)
        command.add_argument('--sinc-backend',choices=('matrix','polyphase'),default='matrix')
    args = parser.parse_args()
    options = RawTestOptions(guards=getattr(args,'guards',4), interpolation=args.interpolation,
        sinc_half_width=args.sinc_half_width, beta=args.beta, sinc_oversampling=args.sinc_oversampling,
        sinc_backend=args.sinc_backend,
        scene_epoch_s=(None if getattr(args,'legacy_scene_clock',False) else getattr(args,'scene_epoch_s',1.)),
        save_coarse=getattr(args,'save_coarse',False))
    if args.command == 'generate':
        sar_raw_test(args.config,args.output,args.ocean,args.reuse_ocean,options=options)
    else:
        print(reinterpolate(args.inputs,args.template,args.output,options=options,
                            workers=args.workers,cfg_file=args.config))


if __name__ == '__main__':
    main()
