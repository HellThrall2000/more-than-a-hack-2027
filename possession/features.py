"""L3 -- the per-frame feature matrix (PLAN.md 5.4).

One code path serves the real match and simulated external matches, so that a model trained on one
can be run on the other. Features read the ball through `src`/`sigma`: a ball filled by a carrier
search must not look like an observation. Smart Tagging is deliberately NOT a column here -- it
exists only for this match -- and enters at the emission stage as a separate calibrated term.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .ball import BallState
from .calm import calm_probability
from .config import Config
from .players import nearest_by_team, velocity
from .restarts import RESTART_TYPES, formation_scores
from .util import rolling_nanmean

CONTEXT_COLS = ("margin", "ball_speed", "walk_frac", "oob", "stationary")
CONTEXT_HALF = (5, 25, 75)      # +/- 0.2 s, 1 s, 3 s


@dataclasses.dataclass
class FeatureSet:
    F: np.ndarray
    names: list

    def __post_init__(self):
        self._index = {name: i for i, name in enumerate(self.names)}

    def col(self, name):
        return self.F[:, self._index[name]]


def _team_distance_stats(players_xy, slot_team, ball):
    """Nearest and second-nearest distance per team, and counts within 1, 2, 5 m."""
    d = np.hypot(players_xy[:, :, 0] - ball[:, None, 0], players_xy[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    out = {}
    for t in (0, 1):
        dt = np.where(slot_team[None, :] == t, d, np.inf)
        part = np.partition(dt, 1, axis=1)[:, :2]
        out["d1", t] = np.where(np.isfinite(part.min(axis=1)), part.min(axis=1), np.nan)
        out["d2", t] = np.where(np.isfinite(part.max(axis=1)), part.max(axis=1), np.nan)
        for r in (1, 2, 5):
            out["n%d" % r, t] = (dt < r).sum(axis=1).astype(np.float32)
    return out


def build_features(ball: BallState, players_xy, desc, slot_team, attack, blocks, cfg: Config):
    """Returns (FeatureSet, aux) with aux holding arrays the later stages need by name."""
    n = len(ball.xy)
    fps, length, width = cfg.fps, cfg.pitch_length, cfg.pitch_width
    cols = {}

    st = _team_distance_stats(players_xy, slot_team, ball.xy)
    dist, slot = nearest_by_team(players_xy, slot_team, ball.xy)
    pv = velocity(players_xy, cfg)
    for t, tag in ((0, "h"), (1, "a")):
        cols["d_" + tag] = st["d1", t]
        cols["d2_" + tag] = st["d2", t]
        for r in (1, 2, 5):
            cols["n%d_%s" % (r, tag)] = st["n%d" % r, t]
        rv = np.hypot(*(ball.vel - pv[np.arange(n), slot[:, t]]).T)
        cols["rv_" + tag] = rv
    cols["margin"] = cols["d_a"] - cols["d_h"]

    tf = ball.touches["touch_frame"]
    since = np.full(n, 10.0, dtype=np.float32)
    sign = np.zeros(n, dtype=np.float32)
    if len(tf):
        idx = np.arange(n)
        k = np.searchsorted(tf, idx, side="right") - 1
        has = k >= 0
        since[has] = np.minimum((idx[has] - tf[k[has]]) / float(fps), 10.0)
        sign[has] = np.where(ball.touches["touch_team"][k[has]] == 0, 1.0, -1.0)
    cols["since_touch"], cols["touch_team"] = since, sign

    speed = ball.speed
    acc = np.full(n, np.nan, dtype=np.float32)
    acc[1:-1] = np.abs(speed[2:] - speed[:-2]) * (fps / 2.0)
    v = ball.vel
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (v[:-2] * v[2:]).sum(axis=1) / (np.hypot(*v[:-2].T) * np.hypot(*v[2:].T))
    turn = np.full(n, np.nan, dtype=np.float32)
    turn[1:-1] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    cols.update(ball_speed=speed, ball_acc=acc, ball_turn=turn, src=ball.src.astype(np.float32),
                sigma=ball.sigma, gap_len=ball.gaps.gap_len.astype(np.float32),
                since_seen=ball.gaps.since_seen.astype(np.float32),
                until_seen=ball.gaps.until_seen.astype(np.float32))

    # position in the home team's attacking frame, per half
    bx_att = np.full(n, np.nan, dtype=np.float32)
    for half, a, b in blocks:
        home_dir = attack[half][0]
        bx = ball.xy[a:b, 0] / length
        bx_att[a:b] = bx if home_dir > 0 else 1.0 - bx
    by = ball.xy[:, 1]
    cols["bx_att"] = bx_att
    cols["by"] = by
    cols["dist_line"] = np.minimum.reduce([ball.xy[:, 0], length - ball.xy[:, 0], by, width - by])
    cols["oob"] = ball.oob.astype(np.float32)
    cols["stationary"] = ball.stationary.astype(np.float32)
    cols["ctrl_h"] = (ball.ctrl_team == 0).astype(np.float32)
    cols["ctrl_a"] = (ball.ctrl_team == 1).astype(np.float32)
    cols["carrier_h"] = (ball.carrier_team == 0).astype(np.float32)
    cols["carrier_a"] = (ball.carrier_team == 1).astype(np.float32)

    cols["mean_speed_h"], cols["mean_speed_a"] = desc["mean_speed"][:, 0], desc["mean_speed"][:, 1]
    cols["walk_frac"] = desc["walk_frac"]
    cols["calm"] = calm_probability(desc["mean_speed"], blocks, cfg)
    cols["spread_x_h"], cols["spread_x_a"] = desc["spread"][:, 0, 0], desc["spread"][:, 1, 0]
    cols["cent_att_h"] = np.full(n, np.nan, dtype=np.float32)
    cols["cent_att_a"] = np.full(n, np.nan, dtype=np.float32)
    for half, a, b in blocks:
        for t, tag in ((0, "h"), (1, "a")):
            cx = desc["centroid"][a:b, t, 0] / length
            cols["cent_att_" + tag][a:b] = cx if attack[half][t] > 0 else 1.0 - cx
    wf = cols["walk_frac"]
    trend = np.full(n, np.nan, dtype=np.float32)
    trend[25:-25] = wf[50:] - wf[:-50]
    cols["walk_trend"] = trend

    slow = ball.stationary | (np.nan_to_num(speed, nan=99.0) < 1.5)
    d_star = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    rest = formation_scores(ball.xy, slow, d_star, players_xy, slot_team, cfg)
    for i, name in enumerate(RESTART_TYPES):
        cols["rs_" + name] = rest[:, i]

    for name in CONTEXT_COLS:
        for half in CONTEXT_HALF:
            cols["%s_ctx%d" % (name, half)] = rolling_nanmean(cols[name], half)

    names = list(cols)
    F = np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names])
    aux = {"dist": dist, "slot": slot, "d_star": d_star, "rest": rest, "slow": slow}
    return FeatureSet(F, names), aux
