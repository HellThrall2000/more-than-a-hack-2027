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

TAG_MIN_REF = 10            # reference frames an interval needs to be scored
TAG_MIN_INTERVALS = 5       # scored intervals a code needs to get any weight

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


def kickoff_teams(tags):
    """{half: 0 home / 1 away} for the start-of-half kick-offs, from 'Kick Off (Start)' set pieces.

    Law 8: the team that did not kick off the first half kicks off the second, so one known half
    gives the other. Halves whose tags disagree with each other or with that rule are left out.
    """
    seen = {}
    for tag in tags:
        if (tag.get("code") or "").strip().upper() != SET_PIECE:
            continue
        if "kick off (start)" not in str(tag["tags"].get("Type", "")).lower():
            continue
        team = team_index(tag["tags"].get("Team"))
        half = str(tag["tags"].get("Half", "")).strip()[:1]
        if team is not None and half in ("1", "2"):
            seen.setdefault(int(half), set()).add(team)
    known = {h: next(iter(t)) for h, t in seen.items() if len(t) == 1}
    if len(known) == 2 and known[1] == known[2]:
        return {}               # contradicts Law 8: trust neither
    for h in (1, 2):
        if h in known and 3 - h not in known:
            known[3 - h] = 1 - known[h]
    return known


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


def calibrate(tags, reference, cfg: Config):
    """{code: weight} and {code: (agree, disagree)} against a reference team per frame (-1 = none).
    No ground truth involved.

    cfg.tag_calib = "interval" (default): `reference` is the control-episode team. Each tag
    interval with at least TAG_MIN_REF reference frames scores the share of them that agree, and a
    code's accuracy is the mean over its intervals (+1/+2 smoothing), so a long interval counts
    once and not once per frame. Codes with fewer than TAG_MIN_INTERVALS usable intervals get no
    weight. counts holds (agreeing, disagreeing) intervals.

    cfg.tag_calib = "silver": the earlier per-frame accuracy on silver frames (silver_team).
    """
    per_code = {}
    for tag in tags:
        team = possessing_team(tag)
        if team is None:
            continue
        code = tag["code"].strip().upper()
        seg = reference[max(0, tag["start_frame"]):tag["end_frame"]]
        seg = seg[seg >= 0]
        per_code.setdefault(code, []).append((int((seg == team).sum()), int((seg != team).sum())))
    weights, counts = {}, {}
    for code, rows in per_code.items():
        if cfg.tag_calib == "silver":
            agree, disagree = sum(r[0] for r in rows), sum(r[1] for r in rows)
            counts[code] = (agree, disagree)
            if agree + disagree < 20:
                weights[code] = 0.0
                continue
            acc = (agree + 1.0) / (agree + disagree + 2.0)
        else:
            shares = [a / float(a + d) for a, d in rows if a + d >= TAG_MIN_REF]
            counts[code] = (sum(s > 0.5 for s in shares), sum(s <= 0.5 for s in shares))
            if len(shares) < TAG_MIN_INTERVALS:
                weights[code] = 0.0
                continue
            acc = (sum(shares) + 1.0) / (len(shares) + 2.0)
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
                # Overlapping phase codes (BALL IN FINAL THIRD + BALL IN THE BOX + PLAYERS IN THE
                # BOX ...) describe one situation, not independent evidence: by default a team's
                # tag evidence is its strongest tag, not the sum, so stacked codes cannot outvote
                # the tracking.
                w = weights.get(code, 0.0) * r
                if cfg.tag_pool == "sum":
                    poss[a:b, team] += w
                else:
                    poss[a:b, team] = np.maximum(poss[a:b, team], w)
    return {"poss": poss, "setpiece": setpiece, "goal": goal}
