"""The ball scorecard (PLAN.md 5.3.7): how accurately does the gap filler track the ball?

No ground truth is needed. Visible stretches of this match's own ball track are masked using the
gap-length distribution of the same track, the filler reconstructs them, and the result is compared
with the held-out positions. It measures the ball layer in isolation, so a change to gap handling
can be judged without touching the possession decoder.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..players import nearest_by_team
from ..util import runs
from . import clean as _clean
from . import smooth as _smooth
from .gaps import SRC_CARRIED, SRC_NAMES, reconstruct

BUCKETS = ((3, 8), (9, 25), (26, 75), (76, 10 ** 6))


def mask_windows(visible, lengths, rng, n_windows):
    """Windows [(start, length)] drawn inside visible runs, lengths from the empirical gaps."""
    starts, stops = runs(visible)
    windows = []
    for _ in range(n_windows * 4):
        if len(windows) >= n_windows:
            break
        length = int(lengths[rng.randint(len(lengths))])
        if length < BUCKETS[0][0]:
            continue
        k = rng.randint(len(starts))
        room = (stops[k] - starts[k]) - length - 24
        if room <= 0:
            continue
        windows.append((int(starts[k] + 12 + rng.randint(room + 1)), length))
    return windows


def run(ball_xy, players_xy, slot_team, blocks, cfg: Config, n_windows=300, seed=0):
    """Aggregate error statistics of gap filling on masked, previously visible frames."""
    rng = np.random.RandomState(seed)
    truth = _clean.clean(ball_xy, cfg)
    visible = ~np.isnan(truth[:, 0])
    g0, g1 = runs(~visible)
    lengths = (g1 - g0)[(g1 - g0) >= BUCKETS[0][0]]
    if not len(lengths):
        return {"error": "no gaps to model"}

    windows = mask_windows(visible, lengths, rng, n_windows)
    masked = truth.copy()
    held = np.zeros(len(truth), dtype=bool)
    wlen = np.zeros(len(truth), dtype=np.int32)
    for s, length in windows:
        masked[s:s + length] = np.nan
        held[s:s + length] = True
        wlen[s:s + length] = length

    pos, vel, observed, interp = _smooth.smooth_ball(masked, cfg)
    gaps = reconstruct(pos, vel, observed & ~np.isnan(pos[:, 0]), interp, players_xy, slot_team,
                       blocks, cfg)
    err = np.hypot(*(gaps.xy - truth).T)

    out = {"n_windows": len(windows), "n_frames": int(held.sum()), "by_length": {}, "by_src": {}}
    for lo, hi in BUCKETS:
        sel = held & (wlen >= lo) & (wlen <= hi) & ~np.isnan(err)
        if sel.any():
            out["by_length"]["%d-%s" % (lo, hi if hi < 10 ** 6 else "inf")] = {
                "frames": int(sel.sum()), "median_m": float(np.median(err[sel])),
                "p90_m": float(np.percentile(err[sel], 90))}
    for s, name in enumerate(SRC_NAMES):
        sel = held & (gaps.src == s) & ~np.isnan(err)
        if sel.any():
            out["by_src"][name] = {"frames": int(sel.sum()), "median_m": float(np.median(err[sel])),
                                   "mean_sigma_m": float(np.nanmean(gaps.sigma[sel]))}

    # carrier-team accuracy: the team of the true nearest player, when one is within 3 m
    sel = held & (gaps.src == SRC_CARRIED) & ~np.isnan(truth[:, 0])
    if sel.any():
        dist, _ = nearest_by_team(players_xy[sel], slot_team, truth[sel])
        d = np.where(np.isnan(dist), np.inf, dist)
        near = d.min(axis=1) < 3.0
        true_team = d.argmin(axis=1)
        if near.any():
            out["carrier_team_accuracy"] = float((gaps.carrier_team[sel][near] == true_team[near]).mean())
            out["carrier_frames_scored"] = int(near.sum())
    return out
