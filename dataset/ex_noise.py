"""Horizon-only Gaussian noise for exogenous future features."""

from __future__ import annotations

import numpy as np


def add_horizon_ex_noise(
    xf: np.ndarray,
    x_size: int,
    scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise to the forecast-horizon slice of exogenous features.

    Noise std per channel is ``scale * std(history)``, using only the history
    window ``xf[:x_size]`` (no future leakage into the standard deviation).

    Parameters
    ----------
    xf : np.ndarray
        Shape ``[L+H, C]`` — history then horizon exogenous values.
    x_size : int
        History length ``L``. Noise is applied only to ``xf[x_size:]``.
    scale : float
        Multiplier on per-channel history std.
    rng : np.random.Generator, optional
        RNG for reproducibility; a fresh default generator is used if omitted.

    Returns
    -------
    np.ndarray
        Copy of ``xf`` with noise added to the horizon slice.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = xf.copy()
    hist = out[:x_size]
    sigma = scale * np.std(hist, axis=0, ddof=0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    # zero-std channels get no noise
    noise = rng.normal(loc=0.0, scale=1.0, size=out[x_size:].shape) * sigma
    out[x_size:] = out[x_size:] + noise
    return out
