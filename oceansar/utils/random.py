"""Utilities for reproducible, independent simulation random streams."""

from contextlib import contextmanager

import numpy as np


_STREAM_IDS = {
    "surface": 0,
    "bragg_positive": 1,
    "bragg_negative": 2,
    "sar_noise_hh": 3,
    "sar_noise_vv": 4,
    "dopsca_thermal_noise": 5,
    "dopsca_phase_noise": 6,
}


def get_sim_seed(cfg):
    """Return the optional non-negative integer seed from ``cfg.sim``."""
    seed = getattr(cfg.sim, "seed", None)
    if seed is None:
        return None
    if isinstance(seed, str) and seed.strip().lower() in {
            "none", "auto", "random"}:
        return None
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("sim.seed must be a non-negative integer or None")
    if isinstance(seed, (float, np.floating)) and not seed.is_integer():
        raise ValueError("sim.seed must be a non-negative integer or None")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sim.seed must be a non-negative integer or None"
        ) from exc
    if seed < 0:
        raise ValueError("sim.seed must be a non-negative integer or None")
    return seed


def _seed_sequence(seed, stream):
    try:
        stream_id = _STREAM_IDS[stream]
    except KeyError as exc:
        raise ValueError("Unknown random stream: %s" % stream) from exc
    return np.random.SeedSequence([seed, stream_id])


def random_generator(seed, stream):
    """Create an independent generator for a named simulation stream."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(_seed_sequence(seed, stream))


@contextmanager
def legacy_random_seed(seed, stream):
    """Temporarily seed legacy ``np.random`` code for one named stream."""
    if seed is None:
        yield
        return

    state = np.random.get_state()
    legacy_seed = _seed_sequence(seed, stream).generate_state(
        1, dtype=np.uint32
    )[0]
    np.random.seed(legacy_seed)
    try:
        yield
    finally:
        np.random.set_state(state)
