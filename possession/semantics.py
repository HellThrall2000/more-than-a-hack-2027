"""Week-1 semantics study (SITUATION.md section 8). Aggregate offsets only -- keep the output local.

    python -m possession.semantics [--gt public|PATH]

Where does the ground truth put each boundary, relative to the events our detectors fire on? For
every true transition it finds the nearest event of the matching kind within +/- SEARCH frames and
reports the signed offset, event minus truth (negative: the event comes first), as a count, median
and range per kind of event:

  Q1  alive -> D   out-of-play onset, ball at rest with players walking, whistle, calm onset,
                   SET PIECES start
  Q2  D -> alive   ball leaves rest, out-of-play end, calm end, SET PIECES start
  Q3  H <-> A      control onset and touch by the team that gains the ball
  Q4  stoppages    SET PIECES Type around each true D run, and which team restarts

It answers what the definition is. It must not choose between versions or tune thresholds: the
public window has 17 possessions, which cannot rank anything (EVALUATION.md).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import log as plog
from .ball.events import ctrl_episodes
from .candidates import rest_leaves, stoppage_runs
from .config import DEFAULT, Config
from .policy import metrics
from .tags import GOAL, SET_PIECE, team_index

log = plog.get(__name__)
SEARCH = 75          # frames: 3 s either side of the true boundary
H, A, D = 0, 1, 2
NAMES = ("H", "A", "D")


def signed_nearest(events, targets, search=SEARCH):
    """Offset (event - target) to the nearest event, per target; NaN where none is within search.
    A tie goes to the earlier event."""
    events = np.sort(np.asarray(events, dtype=np.int64))
    out = np.full(len(targets), np.nan)
    if not len(events):
        return out
    for i, t in enumerate(targets):
        k = int(np.searchsorted(events, t))
        d = events[max(0, k - 1):k + 1] - t
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) <= search:
            out[i] = d[j]
    return out


def _line(name, offsets):
    found = offsets[~np.isnan(offsets)]
    head = "    {:<30} found {:>2}/{:<2}".format(name, len(found), len(offsets))
    if not len(found):
        return head
    return head + "  median {:+5.0f} fr  range {:+.0f} .. {:+.0f}".format(
        np.median(found), found.min(), found.max())


def _code(tag):
    return (tag.get("code") or "").strip().upper()


def detector_events(details, tags, cfg: Config):
    """Frames at which each detector fires, from the internals of one predict(details=True) run."""
    ball = details["ball"]
    stops = stoppage_runs(ball, details["features"], cfg)
    eps = ctrl_episodes(ball.ctrl_team)
    tf, tt = ball.touches["touch_frame"], ball.touches["touch_team"]
    setpiece = np.array([t["start_frame"] for t in tags if _code(t) == SET_PIECE], dtype=np.int64)
    goal = np.array([t["start_frame"] for t in tags if _code(t) == GOAL], dtype=np.int64)
    return {"stop": {"out-of-play onset": stops["oob"][0],
                     "at rest + walking onset": stops["still"][0],
                     "whistle (deceleration) onset": stops["whistle"][0],
                     "calm onset (players stop)": stops["calm"][0],
                     "SET PIECES tag start": setpiece,
                     "GOALS tag start": goal},
            "restart": {"ball leaves rest": [b for _, b in rest_leaves(ball, details["aux"], cfg)],
                        "out-of-play end": stops["oob"][1],
                        "calm end (players go again)": stops["calm"][1],
                        "SET PIECES tag start": setpiece},
            "turn": {"control onset, new team": {t: [a for a, _, k in eps if k == t] for t in (H, A)},
                     "touch, new team": {t: tf[tt == t] for t in (H, A)}},
            "touches": (tf, tt)}


def report(gt, details, tags, cfg: Config = DEFAULT):
    """The semantics text for one ground truth. `gt` is an int8 code array (0 H, 1 A, 2 D, 3 X)."""
    pm = metrics()
    runs = pm.valid_runs(gt)
    tr = pm.transitions(gt, runs)
    ev = detector_events(details, tags, cfg)
    out = []
    p = out.append
    p("semantics study -- aggregate offsets only; keep this output on your machine")
    p("offset = detector event - true boundary, in frames (25 = 1 s); negative = event first; "
      "search +/- {} fr".format(SEARCH))

    stop_f = np.array([f for f, a, b in tr if b == D], dtype=np.int64)
    p("\nQ1  alive -> D  (n={})  does D start at the line crossing, or at the whistle?".format(len(stop_f)))
    for name, frames in ev["stop"].items():
        p(_line(name, signed_nearest(frames, stop_f)))

    restart_f = np.array([f for f, a, b in tr if a == D], dtype=np.int64)
    p("\nQ2  D -> alive  (n={})  does alive start on the strike?".format(len(restart_f)))
    for name, frames in ev["restart"].items():
        p(_line(name, signed_nearest(frames, restart_f)))

    turns = [(f, b) for f, a, b in tr if a < D and b < D]
    p("\nQ3  H <-> A  (n={})  first touch of the new team, or after k frames of control?".format(len(turns)))
    for name, by_team in ev["turn"].items():
        offs = np.array([signed_nearest(by_team[team], [f])[0] for f, team in turns])
        p(_line(name, offs))

    p("\nQ4  stoppages")
    tf, tt = ev["touches"]
    types, side = {}, {"same team restarts": 0, "other team restarts": 0,
                       "= opponent of last toucher": 0, "!= opponent of last toucher": 0,
                       "no touch in the 10 s before": 0}
    segs = pm.segments(gt, runs)
    for i, (a, b, k) in enumerate(segs):
        if k != D or i == 0 or i + 1 >= len(segs) or segs[i - 1][1] != a or segs[i + 1][0] != b:
            continue
        before, after = segs[i - 1][2], segs[i + 1][2]
        side["same team restarts" if before == after else "other team restarts"] += 1
        j = int(np.searchsorted(tf, a, side="right")) - 1
        if j < 0 or a - tf[j] > 10 * cfg.fps:
            side["no touch in the 10 s before"] += 1
        else:
            side["= opponent of last toucher" if after == 1 - int(tt[j]) else "!= opponent of last toucher"] += 1
        near = [t for t in tags if _code(t) == SET_PIECE
                and a - 2 * cfg.fps <= t["start_frame"] <= b + 2 * cfg.fps]
        key = near[0]["tags"].get("Type", "?") if near else "no SET PIECES tag"
        types[key] = types.get(key, 0) + 1
    p("    SET PIECES Type around each true D run: {}".format(types or "none"))
    p("    restarting team: {}".format(side))

    kick = {}
    for t in tags:
        if "kick off (start)" in str(t["tags"].get("Type", "")).lower():
            team = team_index(t["tags"].get("Team"))
            kick[t["tags"].get("Half", "?")] = "?" if team is None else NAMES[team]
    p("    Smart Tagging 'Kick Off (Start)' team by half: {}  (Law 8: the two halves differ)".format(
        kick or "none"))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", default="public")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a Config value, e.g. --set emit_tag_weight=0 (repeatable)")
    args = ap.parse_args(argv)
    plog.setup("semantics")

    from .harness import resolve_gt
    from .ingest import load_match
    from .run import predict

    cfg = DEFAULT.with_overrides(args.set)
    if args.set:
        log.info("config overrides %s -> config %s", args.set, cfg.digest())
    pm = metrics()
    gt = pm.label_codes(pm.read_label_csv(resolve_gt(args.gt)))
    with plog.stage("load the match"):
        inputs = load_match(cfg, args.data_dir)
    with plog.stage("predict"):
        _, details = predict(inputs, cfg, details=True)
    log.info("\n%s", report(gt, details, inputs.tags, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
