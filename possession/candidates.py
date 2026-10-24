"""L3 -- candidate boundaries for the decoder (PLAN.md 5.4 "Candidate boundaries").

Generated for recall, scored by the decoder. A candidate carries three bonuses -- for a team
change, for a stoppage starting and for a restart -- because the same frame can be a plausible
boundary of more than one kind. Nothing here is a decision: with no bonus a boundary pays the
decoder's boundary penalty and so needs evidence from the emission to survive.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .ball import BallState
from .ball.events import ctrl_episodes
from .ball.gaps import SRC_OBSERVED
from .config import Config
from .util import keep_long_runs, runs


@dataclasses.dataclass
class Candidates:
    frame: np.ndarray           # (M,) sorted, unique
    turn: np.ndarray            # (M,) bonus for an H<->A change
    stop: np.ndarray            # (M,) bonus for alive -> D
    restart: np.ndarray         # (M,) bonus for D -> alive
    restart_pref: np.ndarray    # (M, 2) extra bonus for the restart going to H / A

    def __len__(self):
        return len(self.frame)


class _Collector:
    def __init__(self):
        self.rows = []      # (frame, turn, stop, restart, pref_h, pref_a)

    def add(self, frame, turn=0.0, stop=0.0, restart=0.0, pref=(0.0, 0.0)):
        self.rows.append((int(frame), turn, stop, restart, pref[0], pref[1]))


def _episode_stream(ball: BallState, cfg: Config):
    """Control episodes from observed frames plus carried-gap carriers, in time order.

    Observed control is full-strength evidence; a carried gap's carrier is weaker.
    """
    eps = [(a, b, t, 1.0) for a, b, t in ctrl_episodes(ball.ctrl_team)]
    carried = ball.carrier_team
    for t in (0, 1):
        for a, b in zip(*runs(carried == t)):
            if b - a >= 10:
                eps.append((int(a), int(b), t, 0.6))
    return sorted(eps)


def stoppage_runs(ball: BallState, fs, cfg: Config):
    """(starts, stops) of each kind of stoppage evidence: the ball out of play, the ball at rest
    with the players walking, both teams slowing down together (a whistle), and the players'
    two-state "calm" read (calm.py) lasting at least calm_min_frames."""
    walk = np.nan_to_num(fs.col("walk_frac"))
    still_walk = ball.stationary & (walk > 0.6)
    with np.errstate(all="ignore"):
        slow_team = np.nan_to_num(np.nanmean(fs.F[:, [fs.names.index("mean_speed_h"),
                                                      fs.names.index("mean_speed_a")]], axis=1), nan=9.0)
    calm = np.nan_to_num(fs.col("calm")) > 0.5
    return {"oob": runs(ball.oob),
            "still": runs(keep_long_runs(still_walk, cfg.rest_min_frames)),
            "whistle": runs(keep_long_runs(slow_team < cfg.whistle_speed, cfg.whistle_min_frames)),
            "calm": runs(keep_long_runs(calm, cfg.calm_min_frames))}


def rest_leaves(ball: BallState, aux, cfg: Config):
    """[(rest_start, leave_frame)]: a ball at rest for rest_min_frames leaves it next to a player."""
    n = len(ball.xy)
    out = []
    for a, b in zip(*runs(keep_long_runs(ball.stationary, cfg.rest_min_frames))):
        if b >= n - 6:
            continue
        after = ball.speed[b:b + 6]
        leaves = bool((~np.isnan(after)).any() and np.nanmax(after) > cfg.rest_leave_speed)
        if leaves and aux["d_star"][b] < 3.0:
            out.append((int(a), int(b)))
    return out


def build_candidates(ball: BallState, fs, aux, blocks, kickoffs, cfg: Config) -> Candidates:
    col = _Collector()

    # -- turnovers: a different team takes control after the previous episode's team ------------
    # The change belongs on the controlling touch (SITUATION.md section 5), which comes before
    # "control" is recognised (the ball close and moving with the player for a few frames), so
    # the boundary goes on the new team's first touch leading into the episode when there is one.
    tf, tt = ball.touches["touch_frame"], ball.touches["touch_team"]
    prev_team = None
    for a, b, t, strength in _episode_stream(ball, cfg):
        if prev_team is not None and t != prev_team:
            s = strength * min(1.0, (b - a) / 12.0)
            col.add(_controlling_touch(tf, tt, a, t, cfg) + cfg.turnover_offset, turn=cfg.dec_bonus_turnover * s)
        prev_team = t
    # touches by the team that is not currently in control
    eps = _episode_stream(ball, cfg)
    starts = np.array([e[0] for e in eps]) if eps else np.zeros(0, dtype=int)
    for f, t, sc in zip(ball.touches["touch_frame"], ball.touches["touch_team"],
                        ball.touches["touch_score"]):
        k = np.searchsorted(starts, f, side="right") - 1
        if k >= 0 and eps[k][2] != t:
            col.add(f, turn=cfg.dec_bonus_touch * float(np.clip(sc / 10.0, 0.3, 1.0)))

    # -- stoppages --------------------------------------------------------------------------------
    stops = stoppage_runs(ball, fs, cfg)
    for a, b in zip(*stops["oob"]):
        col.add(a + cfg.stop_offset, stop=cfg.dec_bonus_stop_oob)
        col.add(b + cfg.restart_offset, restart=cfg.dec_bonus_restart_oob_end, pref=_restart_pref(ball, a, cfg))
    for a in stops["still"][0]:
        col.add(a + cfg.stop_offset, stop=cfg.dec_bonus_stop_still)
    for a in stops["whistle"][0]:
        col.add(a + cfg.stop_offset, stop=cfg.dec_bonus_stop_whistle)
    # the players stop: a stoppage onset, and the moment they start again is a restart candidate
    for a, b in zip(*stops["calm"]):
        col.add(a + cfg.stop_offset, stop=cfg.dec_bonus_stop_whistle)
        col.add(b + cfg.restart_offset, restart=cfg.dec_bonus_restart_oob_end,
                pref=_side_pref(ball, a, cfg))

    # -- restarts: the ball leaves a long rest next to a player -------------------------------------
    formation = aux["rest"].max(axis=1)
    for a, b in rest_leaves(ball, aux, cfg):
        col.add(b + cfg.restart_offset,
                restart=cfg.dec_bonus_restart_leave * (0.5 + 0.5 * float(formation[a:b].max())),
                pref=_side_pref(ball, a, cfg))
    for half, first in (kickoffs or {}).items():
        if first is not None:
            col.add(first, restart=cfg.dec_kickoff_bonus)

    # -- fallback grid: everywhere sparsely, finer inside long ball gaps ------------------------------
    for _, a, b in blocks:
        for f in range(a + cfg.cand_grid, b, cfg.cand_grid):
            col.add(f)
        gl = ball.gaps.gap_len[a:b]
        for g0, g1 in zip(*runs(gl >= cfg.cand_long_gap)):
            for f in range(a + g0 + cfg.cand_grid_gap, a + g1, cfg.cand_grid_gap):
                col.add(f)

    return _merge(col.rows, blocks, cfg)


def _restart_pref(ball: BallState, stop_frame, cfg: Config):
    """After the ball went out, the restart belongs to the opponent of the last toucher."""
    tf, tt = ball.touches["touch_frame"], ball.touches["touch_team"]
    k = np.searchsorted(tf, stop_frame, side="right") - 1
    if k < 0 or stop_frame - tf[k] > 10 * cfg.fps:
        return (0.0, 0.0)
    other = 1 - int(tt[k])
    return (cfg.dec_restart_side, 0.0) if other == 0 else (0.0, cfg.dec_restart_side)


def _controlling_touch(touch_frame, touch_team, onset, team, cfg: Config):
    """The new team's first touch in the run of its touches that leads into a control episode
    starting at `onset`, looking back at most control_lookback frames; `onset` if there is none."""
    lo = int(np.searchsorted(touch_frame, onset - cfg.control_lookback, side="left"))
    hi = int(np.searchsorted(touch_frame, onset, side="right"))
    first = onset
    for k in range(hi - 1, lo - 1, -1):
        if touch_team[k] != team:
            break               # the other team touched it later than this: the run starts after
        first = int(touch_frame[k])
    return first


def _side_pref(ball: BallState, stop_frame, cfg: Config):
    """Restart side by the kind of stoppage (SITUATION.md section 6).

    The ball went out shortly before (out of play within 15 s): throw-in, corner or goal kick,
    which go to the opponent of the last toucher. Otherwise the referee stopped play inside the
    field -- usually a foul on the team that had the ball -- so the free kick goes to the team that
    was in control before the stoppage. (An offside goes the other way; fouls are far more common.)
    """
    lo = max(0, stop_frame - 15 * cfg.fps)
    if ball.oob[lo:stop_frame + 1].any():
        return _restart_pref(ball, stop_frame, cfg)
    for a, b, t in reversed(ctrl_episodes(ball.ctrl_team)):
        if a < stop_frame:
            if stop_frame - b > 10 * cfg.fps:
                break
            return (cfg.dec_restart_side, 0.0) if t == 0 else (0.0, cfg.dec_restart_side)
    return (0.0, 0.0)


def _merge(rows, blocks, cfg: Config) -> Candidates:
    """Inside the blocks only; candidates closer than dec_min_node_gap collapse into the strongest."""
    inside = [r for r in rows
              if any(a + cfg.dec_min_node_gap < r[0] < b - cfg.dec_min_node_gap for _, a, b in blocks)]
    inside.sort(key=lambda r: r[0])
    merged = []
    for r in inside:
        if merged and r[0] - merged[-1][-1][0] <= cfg.dec_min_node_gap:
            merged[-1].append(r)
        else:
            merged.append([r])
    out = np.zeros((len(merged), 6))
    frames = np.zeros(len(merged), dtype=np.int64)
    for i, cluster in enumerate(merged):
        arr = np.array(cluster, dtype=np.float64)
        frames[i] = int(arr[np.argmax(arr[:, 1:4].sum(axis=1)), 0]) if arr[:, 1:4].sum() > 0 \
            else int(arr[0, 0])
        out[i] = arr.max(axis=0)
    # the max frame column is meaningless; frames[] already holds the chosen position
    return Candidates(frames, out[:, 1], out[:, 2], out[:, 3], out[:, 4:6])
