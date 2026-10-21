"""L2 steps 4-5 -- classify every ball gap and fill it as an estimate with an uncertainty.

The ball is missing on ~41% of graded frames, so what happens inside a gap is the backbone of the
system (PLAN.md 5.3). Each frame leaves with a position, a sigma (metres) and a `src` flag, and
downstream code must read the flag: a carrier chosen by dynamic programming is evidence about who
has the ball, never an observation of the ball.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..config import Config
from ..util import runs

SRC_OBSERVED, SRC_INTERP, SRC_CARRIED, SRC_FLIGHT, SRC_UNKNOWN, SRC_NONE = range(6)
SRC_NAMES = ("observed", "interpolated", "carried", "flight", "unknown", "none")


@dataclasses.dataclass
class Gaps:
    xy: np.ndarray            # (N, 2) estimated ball position, metres
    sigma: np.ndarray         # (N,) metres
    src: np.ndarray           # (N,) int8, SRC_*
    carrier_team: np.ndarray  # (N,) int8, -1 unless the frame was filled by `carried`
    carrier_slot: np.ndarray  # (N,) int16
    gap_len: np.ndarray       # (N,) length of the gap this frame sits in, 0 if observed
    since_seen: np.ndarray    # (N,) frames since the last real observation
    until_seen: np.ndarray    # (N,) frames until the next one (the run is offline)


def _edge_velocity(pos, have, idx, step, fps, span=6):
    """Velocity at the end of an observed run from a baseline of up to `span` frames inside it.

    The Savitzky-Golay derivative is least reliable exactly at the run edges, which is where gap
    endpoints are, so a plain finite difference over a longer baseline is used instead. NaN when
    fewer than 3 contiguous frames are available. step=-1: idx ends a run; step=+1: idx starts one.
    """
    for k in range(span, 1, -1):
        j = idx + step * k
        if 0 <= j < len(pos) and have[min(idx, j):max(idx, j) + 1].all():
            lo, hi = (j, idx) if step < 0 else (idx, j)
            return (pos[hi] - pos[lo]) * fps / float(k)
    return np.full(2, np.nan)


def _hermite(p0, p1, v0, v1, w, h):
    w = w[:, None]
    return ((2 * w ** 3 - 3 * w ** 2 + 1) * p0 + (w ** 3 - 2 * w ** 2 + w) * h * v0
            + (-2 * w ** 3 + 3 * w ** 2) * p1 + (w ** 3 - w ** 2) * h * v1)


def _carrier_path(players, bridge, switch_cost):
    """Viterbi over player slots: follow the bridge with as few carrier switches as possible."""
    length, n_slots = players.shape[0], players.shape[1]
    cost = np.hypot(players[:, :, 0] - bridge[:, None, 0], players[:, :, 1] - bridge[:, None, 1])
    cost = np.where(np.isnan(cost), 1e3, cost)
    acc = np.empty((length, n_slots))
    back = np.zeros((length, n_slots), dtype=np.int64)
    acc[0] = cost[0]
    stay = np.arange(n_slots)
    for i in range(1, length):
        prev = acc[i - 1]
        best = int(prev.argmin())
        switch = prev[best] + switch_cost
        take = switch < prev
        acc[i] = np.where(take, switch, prev) + cost[i]
        back[i] = np.where(take, best, stay)
    path = np.zeros(length, dtype=np.int64)
    s = int(acc[-1].argmin())
    for i in range(length - 1, -1, -1):
        path[i] = s
        s = back[i, s]
    return path, float(cost[np.arange(length), path].mean())


def reconstruct(pos, vel, observed, interp, players_xy, slot_team, blocks, cfg: Config) -> Gaps:
    """Fill every gap inside the graded blocks. `pos`/`vel` come from smooth_ball."""
    n = len(pos)
    have = ~np.isnan(pos[:, 0])
    est = pos.copy()
    sigma = np.full(n, np.nan)
    src = np.full(n, SRC_NONE, dtype=np.int8)
    carrier_team = np.full(n, -1, dtype=np.int8)
    carrier_slot = np.full(n, -1, dtype=np.int16)
    gap_len = np.zeros(n, dtype=np.int32)
    since = np.zeros(n, dtype=np.int32)
    until = np.zeros(n, dtype=np.int32)

    src[observed] = SRC_OBSERVED
    src[interp] = SRC_INTERP
    sigma[observed] = cfg.sigma_observed
    sigma[interp] = 2 * cfg.sigma_observed

    for _, a, b in blocks:
        for g0, g1 in zip(*runs(~have[a:b])):
            g0, g1 = g0 + a, g1 + a
            length = g1 - g0
            has_prev, has_next = g0 > a, g1 < b
            k = np.arange(1, length + 1)
            near = np.minimum(k, length + 1 - k)
            gap_len[g0:g1] = length
            since[g0:g1] = k if has_prev else length + 1
            until[g0:g1] = length + 1 - k if has_next else length + 1

            if not has_prev and not has_next:
                est[g0:g1] = np.nanmean(players_xy[g0:g1], axis=(0, 1))
                sigma[g0:g1], src[g0:g1] = 30.0, SRC_UNKNOWN
                continue
            if not (has_prev and has_next):
                est[g0:g1] = pos[g0 - 1] if has_prev else pos[g1]
                sigma[g0:g1] = np.minimum(3.0 + 0.2 * length, 25.0)
                src[g0:g1] = SRC_UNKNOWN
                continue

            p0, p1 = pos[g0 - 1], pos[g1]
            v0 = _edge_velocity(pos, have, g0 - 1, -1, cfg.fps)
            v1 = _edge_velocity(pos, have, g1, +1, cfg.fps)
            w = k / float(length + 1)
            linear = p0 + w[:, None] * (p1 - p0)
            s0 = 0.0 if np.isnan(v0).any() else float(np.hypot(*v0))
            s1 = 0.0 if np.isnan(v1).any() else float(np.hypot(*v1))
            bridge_speed = float(np.hypot(*(p1 - p0))) * cfg.fps / (length + 1)

            if length <= cfg.gap_short:
                if np.isnan(v0).any() or np.isnan(v1).any():
                    est[g0:g1] = linear
                else:
                    est[g0:g1] = _hermite(p0, p1, v0, v1, w, (length + 1) / float(cfg.fps))
                sigma[g0:g1] = cfg.sigma_observed + 0.08 * near
                src[g0:g1] = SRC_INTERP
            elif max(s0, s1, bridge_speed) > cfg.gap_flight_speed:
                est[g0:g1] = linear
                sigma[g0:g1] = np.minimum(0.5 + 0.15 * near, 8.0)
                src[g0:g1] = SRC_FLIGHT
            elif length <= cfg.gap_long:
                path, mean_cost = _carrier_path(players_xy[g0:g1], linear, cfg.gap_carrier_switch)
                if mean_cost > 8.0:           # nobody is anywhere near the bridge
                    est[g0:g1] = linear
                    sigma[g0:g1] = 4.0 + 0.1 * length
                    src[g0:g1] = SRC_UNKNOWN
                else:
                    est[g0:g1] = players_xy[np.arange(g0, g1), path]
                    sigma[g0:g1] = 1.0 + 0.5 * mean_cost + 0.02 * length
                    src[g0:g1] = SRC_CARRIED
                    carrier_slot[g0:g1] = path
                    carrier_team[g0:g1] = slot_team[path]
            else:
                est[g0:g1] = linear
                sigma[g0:g1] = np.minimum(3.0 + 0.15 * near, 25.0)
                src[g0:g1] = SRC_UNKNOWN
    return Gaps(est, sigma, src, carrier_team, carrier_slot, gap_len, since, until)


def out_of_view(est, homography, valid, cfg: Config, image=(1920, 1080), margin=0.0):
    """True where the estimated ball position projects outside the tactical image.

    Needs the homography. Used as a feature on gap frames: a ball that is out of the camera's view
    is an explanation for its absence that is not an occlusion.
    """
    norm = est / np.array([cfg.pitch_length, cfg.pitch_width])
    h = homography
    with np.errstate(invalid="ignore", divide="ignore"):
        den = h[:, 2, 0] * norm[:, 0] + h[:, 2, 1] * norm[:, 1] + h[:, 2, 2]
        px = (h[:, 0, 0] * norm[:, 0] + h[:, 0, 1] * norm[:, 1] + h[:, 0, 2]) / den
        py = (h[:, 1, 0] * norm[:, 0] + h[:, 1, 1] * norm[:, 1] + h[:, 1, 2]) / den
        outside = ((px < -margin) | (px > image[0] + margin)
                   | (py < -margin) | (py > image[1] + margin))
    return outside & np.asarray(valid, dtype=bool)
