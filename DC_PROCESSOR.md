# Single-channel Doppler centroid processing

`oceansar.radarsim.dc_processor` reads the complex SLC in an OceanSAR
`ProcFile`. It processes one selected channel or every channel independently with
`[dc] channel=all`, estimating each polarization with CDE and SDE. It does not require an ocean
state file, a second channel, or an InSAR/ATI L1b product.

From the OceanSAR checkout, in the `oceansar` environment:

```sh
python -m oceansar.radarsim.dc_processor \
  -p /path/to/proc_data.nc -o /path/to/dc_data.nc \
  --resolution 100 100 --taper hann --channel 0 --plot
```

The example resolution is configuration, not an algorithm default. Both axes
are in metres: azimuth, ground range. If no configuration file supplies
polarization labels, outputs use `pol0`, `pol1`, etc. Use `--polarization 0`
to select an index, or `--polarization vv` with a configuration containing
`[mode] pol=DP`. With no selection, process all polarizations in the chosen
channel.

## Existing simulation configuration

Add a `[dc]` section to an existing simulation configuration:

```ini
[dc]
resolution=[100, 100]
taper=hann
samples_per_resolution=4
channel=all
lag=1
min_pairs=10
min_correlation=0.05
plot_save=True
plot_path=dc_plots/
```

Then run:

```sh
python -m oceansar.radarsim.dc_processor -c par/my_simulation.cfg
```

The CLI uses `[sim] path` and `proc_file`, and writes `[sim] dc_file` when
provided, otherwise `dc_data.nc` below that path. These paths follow the
existing OceanSAR convention and are relative to the working directory.
Explicit CLI input/output paths override the configured paths. The execution notebook honours `[sim] do_dc`, displays DC maps inline, and
saves maps under `[dc] plot_path` relative to the DC output directory.

## Python interface

```python
from oceansar.radarsim.dc_processor import dc_process

dc = dc_process(
    None, '/path/to/proc_data.nc', '/path/to/dc_data.nc',
    resolution=(100., 100.), taper='hann', channel=0, polarization=0,
)
print(dc.dc_mean)
```

Use a configuration path instead of `None` to resolve `[dc]` controls and
polarization names. Explicit non-None keywords override those controls;
use `taper='boxcar'` to override a configured taper with rectangular averaging.
Optional `valid_mask`, `segment_ids`, `nodata`, calibrated `pixel_spacing`,
and explicit `window_samples` pass through to the core. Configured numeric
`window_samples` are converted to integer sample counts. Callable/custom
array tapers require explicit SLC support because they have no defined
resizing rule.

## Resolution and outputs

The numerical core in `oceansar/radarsim/dc_estimator.py` is ported directly
from the general part of `s1d_ewv/processing/dc_estimator.py`. It has no import
or installation dependency on s1d_ewv. Keep the two cores synchronised when
changing the estimator. The port includes weighted CDE, four separately
weighted SDE sign correlations and arcsine reconstruction, finite/valid-pair
masks, segment boundaries, quality diagnostics, and image means.

Resolution is the second-moment effective spatial width:

`R_eff = sqrt(12 * weighted_variance + pixel_spacing**2)`.

Named nonnegative symmetric tapers expand their integer support to meet the
requested resolution. A Hann taper uses approximately 1.597 times the
requested support width. Azimuth SLC support includes extra lag rows. Output
stride rounds down to whole pixels at `R_eff / samples_per_resolution`, capped
at a minimum stride of one input pixel. The processor default is 2 (half the
resolution); the examples use 4 (quarter-resolution spacing). A scalar applies
to both axes, or specify `[AZ, GR]`. Higher sampling adds overlapping estimates,
without changing support or resolution. The output records requested and realised
samples per resolution, spacing, stride and support overlap. The CLI override is
`--samples-per-resolution 4 4`. This is not a strict spatial band-limit guarantee.

The xarray Dataset contains a `polarization` dimension alongside `estimator`,
`azimuth`, and `range`, with `dc`, `dc_mean`, correlations, valid-pair and
weight diagnostics, masked-support moments, and provenance. When valid carrier
frequency metadata is present, `radial_velocity` and `radial_velocity_mean` give
the corresponding equivalent radial velocities in m/s. The full-image
mean pools each valid pair once with unit weights, independently of the local
taper. It is not an average of overlapping local estimates. Files use
h5netcdf; `--plot` also writes DC maps in Hz and equivalent radial m/s per
polarization and estimator to
`dc_plots/` (or the configured `[dc] plot_path`). `dc_process(..., plot_show=True)`
also displays these maps in the notebook.

SLC metadata supplies `az_sampling` when available, otherwise the stored
`prf` sampling rate, and `v_ground`, `rg_sampling`, `inc_angle` in degrees.
Ground spacing is `v_ground/fs_az` and
`c/(2*rg_sampling*sin(inc_angle))`. This constant-incidence approximation can
be overridden with calibrated spacing. Azimuth positions include `az0` when
available; range positions are local ground-range offsets, not absolute
geolocation. Fill masks from both complex components are honoured, and only
the requested channel/polarization raster is read at a time.

## Phase interpretation

DC is the correlation-phase frequency of the stored SLC, modulo `fs_az/lag`.
No implicit basebanding, reramping, ambiguity resolution or processor-response
correction is applied. Equivalent radial velocity uses `v_r = -c*f_D/(2*f0)`;
positive frequency therefore maps to negative radial velocity. This unit conversion
does not remove phase-reference or steering contributions from stored-SLC DC. Input `reramp` and
look-direction metadata, where available, are retained as provenance. This
matters particularly for ScanSAR/steered or deramped products. Provide explicit
continuity labels/masks if an SLC spans discontinuities.

## Validation

```sh
python -m unittest discover -s tests -p 'test_dc_*.py' -v
```

Tests cover both estimators, masks, taper moments and sizing, lag support,
channel/polarization selection, configuration/CLI processing, fill values and
NetCDF output. The initial port was also compared identically against the
s1d_ewv core, and smoke-tested on channel 0 of:

`/Users/paco/LocalDATA/OceanSAR/RL_sim_U8_45_SW1_45_1km/proc_data.nc`.

## Consistent InSAR and ATI resolution

The example configurations specify `[insar] resolution=[AZ_M, GR_M]` in metres.
InSAR derives its `az_ml` and `rg_ml` sample counts from SLC azimuth sampling,
ground speed, range sampling, incidence angle and `ml_win`. The target uses the
same second-moment effective width as DC; it describes the averaging kernel,
not the combined SAR impulse response. Ground-range spacing uses the same local,
constant-incidence approximation as DC. Integer support meets or slightly exceeds
the target, and the raster retains its original pixel spacing.

ATI inherits this averaging from the InSAR product. Configure the target in
`[insar]`; the old `[ati] az_ml/rg_ml` settings did not control InSAR averaging.
Legacy `[insar] az_ml/rg_ml` remain supported when `resolution` is absent.
The console, InSAR NetCDF attributes, ATI text report and `ati_stats.npz`
record window samples, physical support, spacing, taper and achieved resolution.
Ocean-grid reference velocities are averaged at a matching physical width using
independent azimuth and range window sizes. The ATI report identifies its optional
additional boxcar averaging separately from the inherited InSAR averaging.

## DC and ATI comparison

ATI returns an xarray Dataset and writes `[sim] ati_map_file` (default
`ati_maps.nc` beside its text report) with `dc` in Hz, `radial_velocity` in m/s,
phase, means and physical window-centre coordinates. It also retains maps and
coordinates in `ati_stats.npz`. ATI frequency is `phase / (2*pi*tau)`, where
`tau = (b_ati[1]-b_ati[0])/(2*v_orbit)`, using the same negative frequency-to-velocity
conversion as DC. Saved ATI maps include both units; `plot_show=True` displays them.

Partial multilook windows at ROI edges are removed before ATI plots and statistics.
The coordinates include the InSAR ROI origin and the half-pixel kernel-centre shift
for even window lengths. Reprocess InSAR to populate the origin metadata in older
products; legacy files fall back to an assumed scene origin and report that choice.

The notebook displays and saves comparisons under `comparison_plots/` using the
ATI window-centre coverage as the reference frame, with common axes and color scales.
It does not interpolate either raster or alter estimates. DC's global `dc_mean`
continues to pool the complete SLC, whereas ATI means cover its valid interior;
the comparison panels show local maps on their shared domain. ATI follows the
configured InSAR flattening while DC follows stored-SLC phase, so equal units alone
do not make those phase references identical.

## All antenna pairs and lag summaries

ATI computes all `N*(N-1)/2` non-redundant same-polarization antenna pairs using
the existing InSAR coherence LUT. Pair `(i,j)` uses its actual signed separation
`tau=(b_ati[j]-b_ati[i])/(2*v_orbit)`. Each phase is converted to Hz before
averaging. Antenna lag is `j-i`; lag `k` contains `N-k` pairs. Unequal physical
spacing is supported because conversion uses each pair's own separation.

`ati_maps.nc` contains `pair_phase`, `pair_dc`, `pair_radial_velocity`,
`pair_coherence`, `lag_dc`, `lag_radial_velocity`, `lag_valid_pair_count`, and
spatial means/stds for every pair and lag. Coordinates include the channel
indices, antenna lag, baseline, time separation and Doppler ambiguity period.
Lag maps average finite pair estimates equally per pixel; the validity counts
show how many pairs contributed. The text report and notebook display per-pair
and lag-average tables in Hz and m/s. NPZ products also retain the pair and lag
arrays. The legacy `dc`, `radial_velocity`, `phase` and means still select pair 0-1.

ATI figure outputs now show lag-averaged maps and `ATI_lag_summary_<pol>.png`,
which compares individual pair spatial means with lag-average spatial means.
`plot_coh_all` and `plot_ati_phase_all` enable the combined pair matrix diagnostics.
The DC/ATI comparison produces one figure per antenna lag, using the first
selected DC channel (override with `plot_comparison(..., channel=INDEX)`).
Doppler/velocity maps retain symmetric bwr scales, and comparisons share scales across lags.

The example DC configurations use `channel=all`. DC adds a channel dimension,
keeps each channel's maps/means separate, and does not average channels. Numeric
channel selection and its original output dimensions remain supported; the CLI
accepts `--channel all` or a zero-based index. DC's temporal `lag` remains a
separate concept from ATI's antenna lag.

For fixed phase error, Doppler error scales as `1/(2*pi*abs(tau))`; longer lags
increase sensitivity. Their Doppler ambiguity period is also `1/abs(tau)`.
The processor does not resolve ambiguities jointly across antenna lags, so a
longer lag is not an unconditional accuracy improvement.

## Compact pair diagnostics and standard deviations

ATI saves and displays one `ATI_pair_matrix_<pol>.png` per polarization when
amplitude, coherence or phase diagnostics are enabled. Upper-triangle panels show
phase for `(i,j)` with `i<j`, lower-triangle panels show that pair's coherence, and
diagonal panels show each channel's multilooked intensity in dB. Phase uses one
shared symmetric scale with a vertical colorbar; coherence uses one shared scale
with a horizontal colorbar. Its lower limit is the finite minimum minus 0.25,
clamped at zero and rounded down to one decimal, while the upper limit is 1. Diagonal intensities share a 30 dB range shown in the
figure caption. These matrices replace the individual pair and amplitude figures.

ATI pair/lag tables now report `pair_dc_std` and `lag_dc_std` in Hz, along with
velocity standard deviations in m/s. These statistics are also saved in the
NetCDF and NPZ products. DC reports `dc_std` and `radial_velocity_std` for each
channel, polarization and estimator. Standard deviations describe the spatial
spread of finite local estimates (population standard deviation, ddof=0), rather
than uncertainty in the mean. DC's `dc_mean` retains its full-image pooled
correlation estimate while its standard deviation describes the local DC map.

## Notebook provenance and display

`sarsim.ipynb` copies its input configuration, with its original filename and
metadata, into the output directory before running processors. It skips the copy
when source and destination already refer to the same file, matching DopSca.

All DC channel products remain saved. Inline DC display uses one summary figure
per polarization: channel 0 and the arithmetic average of finite channel estimates,
with CDE/SDE in Hz and m/s on shared symmetric scales. The channel-average Dataset
is available as `dc_average`; its means and standard deviations describe the
averaged local maps. The summary PNG is saved as `dc_channel_summary_<pol>.png`
when configured plot saving is enabled.

ATI pair matrices use equal scaling of range and azimuth in metres. Figure height
follows the physical image aspect ratio rather than a fixed slide ratio. Annotation,
tick, axis and colorbar fonts remain doubled, with shared axis labels for readability.

## Shared map coverage

ATI Doppler/velocity output window-centre coverage leads the notebook display
frame in azimuth and ground range. The pair matrix, ATI maps, DC channel summary,
individual saved DC maps and DC/ATI comparisons all use those explicit limits.
Every image uses equal scaling in metres and the same three endpoint/midpoint
ticks. Figure shape cannot stretch the scene or change the limits. The matrix's
figure aspect ratio is derived from the displayed physical extent.

`common_map_bounds(ati)` selects that reference frame; calling the helper with
multiple datasets still provides an intersection when explicitly desired.
`ati_process(..., plot_bounds=BOUNDS)` and plotting helpers' `bounds` arguments
accept dictionaries with `azimuth` and `range` `(minimum, maximum)` pairs in metres.
Requested reference limits remain authoritative when a dataset covers less of
the frame; its native samples remain unchanged, with empty margins where needed.
Full output arrays and statistics remain available in the products. The notebook
postpones DC rendering until ATI coverage is known.
