"""L2 step 3 and 6 -- touches, control episodes, out of play and stationary (PLAN.md 5.3.3, 5.3.6)."""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..players import nearest_by_team, velocity
from ..util import fill_holes, keep_long_runs, nms_peaks, runs


def _shifted_min(x, half):
    """Centred rolling minimum over +/- half frames (inf-safe)."""
    stack = [np.roll(x, k) for k in range(-half, half + 1)]
    out = np.min(stack, axis=0)
    out[:half] = np.inf
    out[-half:] = np.inf
    return out


def nearest_player(ball_pos, players_xy, slot_team):
    """Overall nearest player to the ball: (distance, slot, team); inf/-1 where the ball is NaN."""
    dist, slot = nearest_by_team(players_xy, slot_team, ball_pos)
    d = np.where(np.isnan(dist), np.inf, dist)
    team = d.argmin(axis=1).astype(np.int8)
    idx = np.arange(len(d))
    return d[idx, team], slot[idx, team], team


def control_and_touches(ball_pos, ball_vel, players_xy, slot_team, cfg: Config):
    """Control episodes and touches from reliably observed ball frames.

    ctrl_team (N,) int8: 0/1 while that team controls the ball, -1 otherwise. A touch is a
    velocity change or a fast turn of the ball with a player within touch_dist.
    """
    n = len(ball_pos)
    d_star, slot_star, team_star = nearest_player(ball_pos, players_xy, slot_team)

    pv = velocity(players_xy, cfg)
    rel = np.hypot(*(ball_vel - pv[np.arange(n), slot_star]).T)
    rel = np.where(np.isnan(rel), np.inf, rel)

    close = (d_star < cfg.control_dist) & (rel < cfg.control_rel_speed)
    masks = []
    for t in (0, 1):
        m = close & (team_star == t)
        masks.append(keep_long_runs(fill_holes(m, cfg.control_hole), cfg.control_min_frames))
    ctrl = np.full(n, -1, dtype=np.int8)
    ctrl[masks[0]] = 0
    ctrl[masks[1]] = 1
    both = masks[0] & masks[1]
    ctrl[both] = team_star[both]

    h = cfg.touch_half_window
    v_before, v_after = np.roll(ball_vel, h, axis=0), np.roll(ball_vel, -h, axis=0)
    dv = np.hypot(*(v_after - v_before).T)
    sp_b, sp_a = np.hypot(*v_before.T), np.hypot(*v_after.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (v_before * v_after).sum(axis=1) / (sp_b * sp_a)
    turn = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    fast_turn = (turn > cfg.touch_turn_deg) & (np.minimum(sp_b, sp_a) > cfg.touch_turn_speed)
    score = np.where(fast_turn, np.maximum(dv, cfg.touch_dv + 0.01), dv)
    near = _shifted_min(d_star, 2) < cfg.touch_dist
    score = np.where(near & ~np.isnan(score), score, np.nan)
    frames = nms_peaks(score, cfg.touch_nms, cfg.touch_dv)

    # toucher: the nearest player's team at the closest approach within +/- 2 frames
    teams = np.zeros(len(frames), dtype=np.int8)
    for i, f in enumerate(frames):
        lo, hi = max(0, f - 2), min(n, f + 3)
        teams[i] = team_star[lo + int(np.argmin(d_star[lo:hi]))]
    return {"ctrl_team": ctrl, "touch_frame": frames, "touch_team": teams,
            "touch_score": score[frames] if len(frames) else np.zeros(0),
            "d_star": d_star, "slot_star": slot_star, "team_star": team_star}


def ctrl_episodes(ctrl_team):
    """[(start, stop, team)] of the control episodes, in time order."""
    out = []
    for t in (0, 1):
        for a, b in zip(*runs(ctrl_team == t)):
            out.append((int(a), int(b), t))
    return sorted(out)


def out_of_play(est, src, sigma, vel, cfg: Config):
    """(oob, stationary) boolean arrays from the reconstructed ball (PLAN.md 5.3.6)."""
    from .gaps import SRC_INTERP, SRC_OBSERVED

    length, width = cfg.pitch_length, cfg.pitch_width
    m = cfg.oob_margin
    with np.errstate(invalid="ignore"):
        outside = ((est[:, 0] < -m) | (est[:, 0] > length + m)
                   | (est[:, 1] < -m) | (est[:, 1] > width + m))
    outside &= ~np.isnan(est[:, 0])
    reliable = (src == SRC_OBSERVED) | (src == SRC_INTERP)
    oob = np.zeros(len(est), dtype=bool)
    for a, b in zip(*runs(outside)):
        seen = int(reliable[a:b].sum())
        if seen >= cfg.oob_persist or ((b - a) >= cfg.oob_persist_unobserved
                                       and np.nanmax(sigma[a:b]) < 5.0):
            oob[a:b] = True

    # Stationary = every position in a window of stationary_frames lies within stationary_radius.
    # Velocity is too noisy for this: smoothed ball speed has ~1 m/s of jitter, which is above the
    # speed of a ball that is merely being held still.
    n, w = len(est), cfg.stationary_frames
    still = np.zeros(n, dtype=bool)
    if n >= w:
        from numpy.lib.stride_tricks import sliding_window_view
        x = np.where(reliable, est[:, 0], np.nan)
        y = np.where(reliable, est[:, 1], np.nan)
        wx, wy = sliding_window_view(x, w), sliding_window_view(y, w)
        with np.errstate(invalid="ignore"):
            ok = (~np.isnan(wx).any(axis=1)
                  & ((wx.max(axis=1) - wx.min(axis=1)) < cfg.stationary_radius)
                  & ((wy.max(axis=1) - wy.min(axis=1)) < cfg.stationary_radius))
        cover = np.concatenate(([0], np.cumsum(ok)))
        idx = np.arange(n)
        still = (cover[np.minimum(idx, len(ok) - 1) + 1] - cover[np.maximum(idx - w + 1, 0)]) > 0
    # a ball that vanishes and reappears in the same spot sat still while unseen
    for a, b in zip(*runs(~reliable)):
        if a > 0 and b < n and still[a - 1] and still[b] \
                and np.hypot(*(est[b] - est[a - 1])) < cfg.stationary_radius:
            still[a:b] = True
    stationary = still
    return oob, stationary
