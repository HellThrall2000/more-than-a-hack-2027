"""L1 -- player cleaning and collective state (PLAN.md 5.2). No identities: per-team point sets."""

from __future__ import annotations

import numpy as np

from .config import Config
from .util import runs, rolling_nanmean


def drop_jumps(xy, cfg: Config):
    """NaN out observations whose implied speed from the previous valid frame is impossible."""
    xy = xy.copy()
    step = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1)) * cfg.fps
    bad = step > cfg.player_max_speed
    xy[1:][bad] = np.nan
    return xy


def fill_short_gaps(xy, cfg: Config):
    """Linearly fill per-slot gaps of at most player_gap_fill frames when the ends are plausible."""
    out = xy.copy()
    n = len(xy)
    for s in range(xy.shape[1]):
        missing = np.isnan(xy[:, s, 0])
        for a, b in zip(*runs(missing)):
            if a == 0 or b >= n or b - a > cfg.player_gap_fill:
                continue
            p0, p1 = xy[a - 1, s], xy[b, s]
            if np.hypot(*(p1 - p0)) > cfg.player_max_speed * (b - a + 1) / cfg.fps:
                continue
            w = ((np.arange(a, b) - (a - 1)) / float(b - (a - 1)))[:, None]
            out[a:b, s] = p0 + w * (p1 - p0)
    return out


def drop_duplicates(xy, slot_team, cfg: Config, chunk=2000):
    """Within a team, the later slot of any pair closer than dup_merge_dist is a duplicate."""
    out = xy.copy()
    same = (slot_team[:, None] == slot_team[None, :]) & np.triu(
        np.ones((len(slot_team),) * 2, dtype=bool), 1)
    for lo in range(0, len(xy), chunk):
        block = out[lo:lo + chunk]
        d = np.hypot(block[:, :, None, 0] - block[:, None, :, 0],
                     block[:, :, None, 1] - block[:, None, :, 1])
        close = (d < cfg.dup_merge_dist) & same[None]
        # the second member of each close pair (column index) is the one removed
        block[close.any(axis=1)] = np.nan
    return out


def clean_players(xy, slot_team, cfg: Config):
    return drop_duplicates(fill_short_gaps(drop_jumps(xy, cfg), cfg), slot_team, cfg)


def velocity(xy, cfg: Config, half=2):
    """Central-difference velocity (m/s) over +/- half frames; NaN at the ends and across gaps."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[half:-half] = (xy[2 * half:] - xy[:-2 * half]) * (cfg.fps / (2.0 * half))
    return v


def nearest_by_team(xy, slot_team, points):
    """Distance and slot of the nearest player of each team to `points` (N, 2).

    Returns (dist (N, 2), slot (N, 2)); dist is NaN where a team has no tracked player.
    """
    d = np.hypot(xy[:, :, 0] - points[:, None, 0], xy[:, :, 1] - points[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    dist = np.full((len(xy), 2), np.nan, dtype=np.float32)
    slot = np.zeros((len(xy), 2), dtype=np.int16)
    for t in (0, 1):
        dt = np.where(slot_team[None, :] == t, d, np.inf)
        idx = dt.argmin(axis=1)
        best = dt[np.arange(len(xy)), idx]
        dist[:, t] = np.where(np.isfinite(best), best, np.nan)
        slot[:, t] = idx
    return dist, slot


def attack_direction(xy, slot_team, blocks):
    """{half: (home_dir, away_dir)} with +1 meaning "attacks towards increasing x".

    Over a half the team whose mean x is lower defends the x=0 goal. Teams swap ends at halftime.
    """
    out = {}
    for half, a, b in blocks:
        mean_x = [np.nanmean(xy[a:b, slot_team == t, 0]) for t in (0, 1)]
        home = 1 if mean_x[0] < mean_x[1] else -1
        out[half] = (home, -home)
    return out


def descriptors(xy, speed, slot_team, cfg: Config):
    """Per-frame collective state, no ball needed.

    centroid (N, 2 teams, 2), spread (N, 2, 2) as std of x and y, mean_speed (N, 2),
    walk_frac (N,) share of tracked players walking, n_tracked (N,).
    """
    n = len(xy)
    centroid = np.full((n, 2, 2), np.nan, dtype=np.float32)
    spread = np.full((n, 2, 2), np.nan, dtype=np.float32)
    mean_speed = np.full((n, 2), np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        for t in (0, 1):
            sel = slot_team == t
            centroid[:, t] = np.nanmean(xy[:, sel], axis=1)
            spread[:, t] = np.nanstd(xy[:, sel], axis=1)
            mean_speed[:, t] = np.nanmean(speed[:, sel], axis=1)
        walking = (speed < cfg.walk_speed).astype(np.float32)
        walking[np.isnan(speed)] = np.nan
        walk_frac = np.nanmean(walking, axis=1)
    n_tracked = (~np.isnan(xy[:, :, 0])).sum(axis=1)
    # one second of smoothing: single-frame speed estimates are noisy
    walk_frac = rolling_nanmean(walk_frac, cfg.fps // 2)
    return {"centroid": centroid, "spread": spread, "mean_speed": mean_speed,
            "walk_frac": walk_frac.astype(np.float32), "n_tracked": n_tracked}
