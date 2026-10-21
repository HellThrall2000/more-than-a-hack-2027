"""Small array helpers shared by every layer."""

from __future__ import annotations

import numpy as np


def runs(mask):
    """(starts, stops) of the maximal True runs of a boolean array; stops are exclusive."""
    m = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def fill_holes(mask, max_hole):
    """Turn False holes of at most `max_hole` frames that sit between True frames into True."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    starts, stops = runs(~mask)
    for s, e in zip(starts, stops):
        if s > 0 and e < len(mask) and e - s <= max_hole:
            out[s:e] = True
    return out


def keep_long_runs(mask, min_len):
    """Keep only True runs of at least `min_len` frames."""
    out = np.zeros(len(mask), dtype=bool)
    for s, e in zip(*runs(mask)):
        if e - s >= min_len:
            out[s:e] = True
    return out


def rolling_nanmean(x, half):
    """Centred mean over +/- half frames ignoring NaN; NaN where the window holds no value."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half)
    hi = np.minimum(len(x), idx + half + 1)
    cnt = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[hi] - csum[lo]) / cnt
    out[cnt == 0] = np.nan
    return out


def window_sum(x, half):
    """Centred sum over +/- half frames (NaN counts as zero)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    csum = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(len(x))
    return csum[np.minimum(len(x), idx + half + 1)] - csum[np.maximum(0, idx - half)]


def nms_peaks(score, min_dist, threshold):
    """Indices of local maxima of `score` above `threshold`, at least `min_dist` apart.

    Greedy by descending score, so the strongest peak in a neighbourhood always survives.
    """
    score = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=-np.inf)
    cand = np.flatnonzero(score > threshold)
    if not len(cand):
        return cand
    order = cand[np.argsort(-score[cand], kind="stable")]
    taken = np.zeros(len(score), dtype=bool)
    keep = []
    for i in order:
        if taken[max(0, i - min_dist):i + min_dist + 1].any():
            continue
        taken[i] = True
        keep.append(i)
    return np.array(sorted(keep), dtype=np.int64)
