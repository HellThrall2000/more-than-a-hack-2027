"""L2 step 2 -- Savitzky-Golay smoothing inside visible runs, giving position and velocity."""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..util import fill_holes, runs

_KERNELS = {}


def _kernels(window, order):
    key = (window, order)
    if key not in _KERNELS:
        m = window // 2
        a = np.vander(np.arange(-m, m + 1), order + 1, increasing=True).astype(np.float64)
        pinv = np.linalg.pinv(a)
        _KERNELS[key] = (pinv[0], pinv[1])      # value at the centre, first derivative per frame
    return _KERNELS[key]


def smooth_series(y, window, order):
    """(smoothed, derivative per frame) of one run. Edges use a polynomial fit of the end window."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 1:
        return y.copy(), np.array([np.nan])
    if n < window:
        deg = min(order, n - 1)
        coef = np.polyfit(np.arange(n), y, deg)
        idx = np.arange(n)
        return np.polyval(coef, idx), np.polyval(np.polyder(coef), idx)
    m = window // 2
    c0, c1 = _kernels(window, order)
    sm, dv = np.empty(n), np.empty(n)
    sm[m:n - m] = np.convolve(y, c0[::-1], "valid")
    dv[m:n - m] = np.convolve(y, c1[::-1], "valid")
    idx = np.arange(window)
    coef = np.polyfit(idx, y[:window], order)
    sm[:m] = np.polyval(coef, np.arange(m))
    dv[:m] = np.polyval(np.polyder(coef), np.arange(m))
    coef = np.polyfit(idx, y[-window:], order)
    tail = np.arange(window - m, window)
    sm[n - m:] = np.polyval(coef, tail)
    dv[n - m:] = np.polyval(np.polyder(coef), tail)
    return sm, dv


def smooth_ball(ball_xy, cfg: Config):
    """Returns (pos, vel, observed, interp): smoothed position (m), velocity (m/s), the raw
    observation mask, and the frames filled by tiny-gap interpolation. pos/vel are NaN elsewhere."""
    raw = np.asarray(ball_xy, dtype=np.float64)
    observed = ~np.isnan(raw[:, 0])
    filled = fill_holes(observed, cfg.ball_tiny_gap)
    interp = filled & ~observed

    work = raw.copy()
    for a, b in zip(*runs(interp)):
        w = ((np.arange(a, b) - (a - 1)) / float(b - (a - 1)))[:, None]
        work[a:b] = work[a - 1] + w * (work[b] - work[a - 1])

    pos = np.full_like(work, np.nan)
    vel = np.full_like(work, np.nan)
    for a, b in zip(*runs(filled)):
        for c in (0, 1):
            pos[a:b, c], vel[a:b, c] = smooth_series(work[a:b, c], cfg.sg_window, cfg.sg_order)
    return pos, vel * cfg.fps, observed, interp
