"""Smart Tagging as a soft, calibrated prior (PLAN.md 5.4 and 5.5).

The tags are automatic phases, not possessions: an interval says roughly when a team was doing
something, with fuzzy edges and wrong teams. So each code gets a weight learned from "silver"
frames -- ball seen, one player on it, no opponent near -- where the controlling team is known
without any ground truth. Throw-ins are tagged Team=N/A; they feed the dead-ball evidence but not
the team channel (the restart side is inferred from the last touch instead).
"""

from __future__ import annotations

import numpy as np

from .config import Config

IN_POSSESSION = frozenset([
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX",
    "PLAYERS IN THE BOX", "LONG BALL", "ATTACKING TRANSITION"])
OUT_OF_POSSESSION = frozenset([
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION"])
SET_PIECE, GOAL = "SET PIECES", "GOALS"

HOME_NAMES = ("barcelona", "barça", "barca")
AWAY_NAMES = ("atletic", "athletic")


def team_index(text):
    """0 for the home club, 1 for the away club, None for N/A or anything unrecognised."""
    t = (text or "").strip().lower()
    if any(name in t for name in HOME_NAMES):
        return 0
    if any(name in t for name in AWAY_NAMES):
        return 1
    return None


def possessing_team(tag):
    """Team the tag says has the ball: the tagged team for attacking codes, its opponent for
    defending codes, None for everything else (including N/A)."""
    code = (tag.get("code") or "").strip().upper()
    team = team_index(tag["tags"].get("Team"))
    if team is None:
        return None
    if code in IN_POSSESSION:
        return team
    if code in OUT_OF_POSSESSION:
        return 1 - team
    return None


def _ramp(n, start, stop, ramp):
    """Trapezoid over [start, stop): fades in and out over `ramp` frames, 1 in the middle."""
    start, stop = max(0, start), min(n, stop)
    if stop <= start:
        return start, stop, np.zeros(0)
    idx = np.arange(start, stop)
    r = np.minimum((idx - start + 1) / float(ramp), (stop - idx) / float(ramp))
    return start, stop, np.clip(r, 0.0, 1.0)


def silver_team(dist, reliable, cfg: Config):
    """(N,) team in unambiguous control (-1 elsewhere): reliably seen ball, one player on it."""
    d = np.where(np.isnan(dist), np.inf, dist)
    rows = np.arange(len(d))
    team = d.argmin(axis=1)
    d_own, d_opp = d[rows, team], d[rows, 1 - team]
    ok = reliable & (d_own <= cfg.tag_silver_dist) & (d_opp >= cfg.tag_silver_opp)
    return np.where(ok, team, -1).astype(np.int8)


def calibrate(tags, silver, cfg: Config):
    """{code: weight} and {code: (agree, disagree)} from silver frames, no ground truth involved."""
    counts = {}
    for tag in tags:
        team = possessing_team(tag)
        if team is None:
            continue
        code = tag["code"].strip().upper()
        seg = silver[max(0, tag["start_frame"]):tag["end_frame"]]
        seg = seg[seg >= 0]
        a, d = counts.get(code, (0, 0))
        counts[code] = (a + int((seg == team).sum()), d + int((seg != team).sum()))
    weights = {}
    for code, (agree, disagree) in counts.items():
        if agree + disagree < 20:
            weights[code] = 0.0
            continue
        acc = (agree + 1.0) / (agree + disagree + 2.0)
        weights[code] = float(np.clip(np.log(acc / (1.0 - acc)), 0.0, cfg.tag_weight_cap))
    return weights, counts


def build(tags, n, weights, cfg: Config):
    """Per-frame tag channels: poss (N, 2) weighted team evidence, setpiece (N,), goal (N,)."""
    poss = np.zeros((n, 2), dtype=np.float32)
    setpiece = np.zeros(n, dtype=np.float32)
    goal = np.zeros(n, dtype=np.float32)
    for tag in tags:
        code = (tag.get("code") or "").strip().upper()
        a, b, r = _ramp(n, tag["start_frame"], tag["end_frame"], cfg.tag_ramp)
        if not len(r):
            continue
        if code == SET_PIECE:
            setpiece[a:b] = np.maximum(setpiece[a:b], r)
        elif code == GOAL:
            goal[a:b] = np.maximum(goal[a:b], r)
        else:
            team = possessing_team(tag)
            if team is not None:
                poss[a:b, team] += weights.get(code, 0.0) * r
    return {"poss": poss, "setpiece": setpiece, "goal": goal}
