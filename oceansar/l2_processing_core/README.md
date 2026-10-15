# Ocean Level-2 processing core

This directory is kept byte-for-byte identical in `oceansar.l2_processing_core`
and `s1d_ewv.l2_processing_core`. It currently contains the array-based CDE/SDE
Doppler estimator, geometry-dependent multilook window sizing/reporting,
Welch and quick intensity cross-spectra, and intensity outlier removal.
The estimator supports configurable samples per resolution; its default is 2.

Dependencies are NumPy, SciPy, xarray and DRAMA, already required by both
projects. Cross-spectrum processing reuses DRAMA smoothing and FFT sizing.
Keep imports within this directory relative. Do not import either host project
here. Readers, configuration, plotting and file orchestration stay
outside this directory. Existing estimator import paths re-export this core;
`estimate_dc_s1` remains an s1d_ewv reader adapter.

The shared cross-spectrum kernels take detected intensities, with explicit
pixel spacing (azimuth, range) in metres. The OceanSAR adapter takes those
intensities directly; the s1d_ewv adapters retain their complex-SLC interface
and detect each image exactly once. Wavenumber axes are cycles/m. Welch
averages normalized overlapping tiles; quick_xspec retains its unnormalized
single-window diagnostic convention.

The Welch implementation is based on the newer OceanSAR version: actual
grid spacing replaces the old fixed 5 m, tile support is bounded by image
dimensions, scalar supports are accepted, and bright-tile rejection uses
pre-centering brightness across tiles. Zero-normalization tiles are excluded;
an entirely unusable image raises ValueError. The historical s1d_ewv Welch
`avg` argument remains accepted and unused, as before.

For now, make algorithm changes in the OceanSAR copy, copy this whole directory
to s1d_ewv (excluding `__pycache__`), and run the affected tests in both projects.
From the OceanSAR repository, check that the copies match with:

```sh
diff -ru --exclude=__pycache__ oceansar/l2_processing_core ../s1d_ewv/s1d_ewv/l2_processing_core
```

When extracting a separate package later, move this directory into that package
and update the host imports. No algorithm changes or host readers should be
needed. This directory does not introduce a separately distributed dependency.
