"""Totals-only cross-check of a prediction against Smart Tagging and the audio (safe to share).

    python -m possession.report [--audio] [--data-dir DIR]

Prints counts and rates -- never a frame number, a coordinate or a label sequence -- so the output
can be pasted into a hosted assistant (FAQ.md forbids data extracts, not aggregate scores). Smart
Tagging is about 70% right, so every rate here is a plausibility check, not an accuracy.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import log as plog
from .config import Config, DEFAULT
from .restarts import conceding_team

log = plog.get(__name__)
NEAR_S = 3.0            # a tag or whistle is "at" a stoppage if one starts or ends within this many seconds


def _dead_stretches(segments):
    return [(a, b) for segs in segments for a, b, k in segs if k == 2]


def _first_alive_after(segments, frame):
    """(team, start) of the first possession starting at or after `frame`, else None."""
    for segs in segments:
        for a, b, k in segs:
            if k != 2 and a >= frame:
                return k, a
    return None


def _near(dead, frame, slack):
    return any(a - slack <= frame <= b + slack for a, b in dead)


def summarise(inputs, details, cfg: Config = DEFAULT):
    """A flat dict of counts and rates (no frame-level data)."""
    segments, ball, attack = details["segments"], details["ball"], details["aux"]["attack"]
    dead = _dead_stretches(segments)
    slack = int(NEAR_S * cfg.fps)
    out = {"possessions": sum(1 for segs in segments for _, _, k in segs if k != 2),
           "dead_stretches": len(dead),
           "dead_median_s": round(float(np.median([(b - a) / cfg.fps for a, b in dead])), 1) if dead else None}

    sets = [t for t in inputs.tags if t["code"].lower() == "set pieces"]
    out["set_piece_tags"] = len(sets)
    out["set_piece_tags_at_a_dead_stretch"] = sum(
        _near(dead, t["start_frame"], slack) or _near(dead, t["end_frame"], slack) for t in sets)

    goals = [t for t in inputs.tags if t["code"].lower() == "goals"]
    tally = {"with_a_dead_stretch": 0, "crossing_between_posts_found": 0, "next_team_known": 0,
             "agree": 0, "disagree": 0}
    for t in goals:
        window = [(a, b) for a, b in dead if a - slack <= t["end_frame"] and b + cfg.fps * 240 >= t["start_frame"]]
        if not window:
            continue
        tally["with_a_dead_stretch"] += 1
        a, b = max(window, key=lambda w: w[1] - w[0])
        nxt = _first_alive_after(segments, b)
        half = next((h for h, lo, hi in inputs.blocks if lo <= b < hi), inputs.blocks[0][0])
        conceded = conceding_team(ball.xy, ball.oob, b, attack[half], cfg)
        tally["crossing_between_posts_found"] += int(conceded is not None)
        tally["next_team_known"] += int(nxt is not None)
        if conceded is not None and nxt is not None:
            tally["agree" if nxt[0] == conceded else "disagree"] += 1
    out["goal_tags"] = len(goals)
    out.update({"goal_" + k: v for k, v in tally.items()})

    if inputs.whistles is not None:
        starts = [a for a, _ in dead]
        out["whistles"] = len(inputs.whistles)
        out["whistles_within_3s_of_a_dead_start"] = sum(any(abs(w - s) <= slack for s in starts)
                                                         for w, _ in inputs.whistles)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = DEFAULT.with_overrides(args.set)
    plog.setup("report")
    from .ingest import load_match
    from .run import _whistles, predict
    inputs = load_match(cfg, args.data_dir)
    if args.audio:
        inputs.whistles = _whistles(args.data_dir, cfg)
    _, details = predict(inputs, cfg, details=True)
    for key, value in summarise(inputs, details, cfg).items():
        log.info("%-40s %s", key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
