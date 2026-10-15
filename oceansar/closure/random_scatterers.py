"""Memory-efficient decorrelating complex scatterers from a shared sequence."""

import numpy as np
from numba import njit

from drama import utils


@njit(cache=True)
def _lookup_scatterers(rotated, indices, step, period, out):
    """Fuse offset wrapping, fixed phase selection and the output write."""
    for i in range(indices.size):
        code = np.int64(indices.flat[i])
        offset = code >> 2
        phase = code & 3
        out.flat[i] = rotated[phase * period + (offset + step) % period]


class randomscat_ts:
    """Complex scatterers with a Gaussian temporal spectrum and random offsets.

    Cells reuse one periodic, unit-power sequence. Each cell also has an
    independent, fixed phase in {1, 1j, -1, -1j}; its two phase bits and sequence
    offset share one uint32. Spatial phase randomization removes, in expectation,
    the coherent component shared by cells sampling a finite sequence, preserving
    each cell's temporal lag products exactly. It does not make the reused
    amplitude histories independent.

    Evaluation is stateless: arbitrary times and array slices can be requested
    in any order, without updating a per-cell time series or random state.
    """

    def __init__(self, tau, dim, Fs, N=None, seed=None, rng=None):
        if seed is not None and rng is not None:
            raise ValueError("Pass either seed or rng, not both")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not np.isfinite(Fs) or Fs <= 0:
            raise ValueError("Fs must be finite and positive")
        if rng is None:
            rng = np.random if seed is None else np.random.RandomState(seed)
        if N is None:
            N = int(tau * Fs * 200)
        if int(N) != N or N < 1:
            raise ValueError("N must be a positive integer")
        if N > 2**30:
            raise ValueError("Sequence offset and phase must fit in uint32")
        self.N = 1 if N == 1 else int(utils.optimize_fftsize(int(N), max_prime=5))

        f = np.fft.fftfreq(self.N, 1./Fs)
        a = 1 / tau**2
        spec = np.sqrt(np.pi / a) * np.exp(-np.pi**2 * f**2 / a)
        s_f = rng.normal(size=self.N) + 1j * rng.normal(size=self.N)
        s = np.fft.ifft(s_f * np.sqrt(spec))
        s /= np.sqrt(np.mean(np.abs(s)**2))
        # Four small lookup tables avoid a complex multiplication per cell.
        # Keep the intrinsic zero-frequency coefficient: spatial phases remove
        # its cross-cell coherence without changing the temporal spectrum.
        self._rotated = np.concatenate((s, 1j*s, -s, -1j*s))
        self.s = self._rotated[:self.N]
        # A uniform packed integer draws the offset and phase independently.
        # Use the actual FFT-rounded period, including its final samples.
        integers = getattr(rng, 'integers', None)
        if integers is None:
            integers = rng.randint
        self._packed_indices = integers(0, 4*self.N, size=dim, dtype=np.uint32)
        self.Fs = Fs

    @property
    def ind0(self):
        """Return decoded sequence offsets (a copy, allocated only on access)."""
        return self._packed_indices >> 2

    def scats(self, t, index=None, *, out=None):
        """Evaluate all cells or an indexed subset, optionally into ``out``.

        ``index`` follows NumPy indexing, including integer block selection and
        strided slices. The default result has dtype complex128; a supplied
        writable complex64/complex128 array must match the selected shape and
        is returned.
        Times retain the original truncation to integer sampling steps.
        """
        indices = np.asarray(self._packed_indices if index is None
                             else self._packed_indices[index])
        scalar_output = out is None and indices.ndim == 0
        if out is None:
            out = np.empty(indices.shape, dtype=self.s.dtype)
        elif (not isinstance(out, np.ndarray) or out.shape != indices.shape
              or out.dtype not in (np.dtype('complex64'), np.dtype('complex128'))
              or not out.flags.writeable):
            raise ValueError(
                "out must be a writable complex64/complex128 array matching the selected shape")
        # Reduce in Python before entering the kernel, including negative times
        # and step counts larger than an int64, to keep its integer math bounded.
        step = int(t * self.Fs) % self.N
        _lookup_scatterers(self._rotated, indices, step, self.N, out)
        return out[()] if scalar_output else out
