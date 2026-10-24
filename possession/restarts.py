"""Restart-formation scores (PLAN.md 5.4): how much a frame looks like a given restart."""

from __future__ import annotations

import numpy as np

from .config import Config

RESTART_TYPES = ("kickoff", "corner", "goal_kick", "throw_in", "free_kick", "penalty")
KICKOFF_SPOT_M = 2.5        # a kicker stands within this of the centre spot
KICKOFF_MARGIN_M = 3.0      # ... and the other team is this much further away (Law 8: 9.15 m)


def kickoff_team_from_players(players_xy, slot_team, kickoffs, cfg: Config, window=25):
    """{half: 0 home / 1 away} from who stands on the centre spot in the second before a kick-off.

    The kicking team has a player on the spot; the other team must stay outside the centre circle.
    A half is left out when nobody is on the spot or both teams are close, so a wrong guess never
    gets in.
    """
    spot = np.array([cfg.pitch_length / 2.0, cfg.pitch_width / 2.0])
    out = {}
    for half, frame in (kickoffs or {}).items():
        if frame is None:
            continue
        seg = players_xy[max(0, frame - window):frame + 1]
        d = np.hypot(seg[..., 0] - spot[0], seg[..., 1] - spot[1])
        with np.errstate(all="ignore"):
            near = [np.nanmin(np.where(slot_team[None, :] == t, d, np.nan)) for t in (0, 1)]
        near = [np.inf if not np.isfinite(v) else float(v) for v in near]
        best = int(np.argmin(near))
        if near[best] < KICKOFF_SPOT_M and near[1 - best] - near[best] > KICKOFF_MARGIN_M:
            out[half] = best
    return out


def resolve_kickoff_team(from_tags, from_players):
    """Tags first, then the tracking; Law 8 fills a missing half and vetoes two equal halves."""
    known = dict(from_players or {})
    known.update(from_tags or {})
    if len(known) == 2 and known[1] == known[2]:
        return {}
    for h in (1, 2):
        if h in known and 3 - h not in known:
            known[3 - h] = 1 - known[h]
    return known


def formation_scores(ball, slow, d_star, players_xy, slot_team, cfg: Config):
    """(N, 6) scores in [0, 1] for kick-off, corner, goal kick, throw-in, free kick and penalty.

    `slow` is a boolean: the ball is at rest or nearly so. A restart needs a ball at rest, so every
    score is zero otherwise.
    """
    length, width = cfg.pitch_length, cfg.pitch_width
    bx, by = ball[:, 0], ball[:, 1]
    n = len(ball)
    out = np.zeros((n, len(RESTART_TYPES)), dtype=np.float32)
    slow = slow & ~np.isnan(bx)

    xs = players_xy[:, :, 0]
    with np.errstate(all="ignore"):
        home_left = np.nanmean(xs[:, slot_team == 0] < length / 2, axis=1)
        away_left = np.nanmean(xs[:, slot_team == 1] < length / 2, axis=1)
    split = np.nan_to_num(np.maximum(home_left * (1 - away_left), (1 - home_left) * away_left))
    d_centre = np.hypot(bx - length / 2, by - width / 2)
    out[:, 0] = np.where(slow & (d_centre < 2.5), split, 0.0)

    corner_d = np.min([np.hypot(bx - cx, by - cy) for cx in (0, length) for cy in (0, width)], axis=0)
    out[:, 1] = np.where(slow & (corner_d < 3.0), 1.0, 0.0)

    in_six = ((bx < 6.5) | (bx > length - 6.5)) & (np.abs(by - width / 2) < 10.2)
    out[:, 2] = np.where(slow & in_six, 1.0, 0.0)

    on_touch = ((by < 1.0) | (by > width - 1.0)) & (bx > 1.0) & (bx < length - 1.0)
    out[:, 3] = np.where(slow & on_touch & (d_star < 2.0), 1.0, 0.0)

    d = np.hypot(xs - bx[:, None], players_xy[:, :, 1] - by[:, None])
    ring = (d > 8.5) & (d < 10.5)
    wall = np.max([(ring & (slot_team[None] == t)).sum(axis=1) for t in (0, 1)], axis=0)
    out[:, 4] = np.where(slow & (wall >= 3), np.minimum(1.0, wall / 4.0), 0.0)

    spot = np.min([np.hypot(bx - sx, by - width / 2) for sx in (11.0, length - 11.0)], axis=0)
    out[:, 5] = np.where(slow & (spot < 1.0), 1.0, 0.0)
    return out
