"""Hand-labelled validation windows (V3, PLAN.md section 7): change points in, ground-truth CSV out.

    python -m possession.labels labels_private/win1.txt           # -> labels_private/win1.csv

The labeller watches a window of the tactical video in any player and writes down only the moments
the label changes, one per line, in whichever unit is handy:

    # alice, 2026-10-25, labelled by SITUATION.md
    window 30000 37499          first and last video frame covered (inclusive), one half only
    30000 H                     video frame, then the label from that frame on
    00:20:31.40 D               video time hh:mm:ss.ss as the player shows it (frame = round(s * 25))
    1H 12:05.4 A                match clock: half, then mm:ss (possession.clock)

The first change point must be the window's first frame. Lines are checked: in order, inside the
window, inside one graded half, labels H / A / D only.

The CSV has the submission's shape (every frame, X outside the window), so it is a ground truth for
`python -m possession.harness <prediction>.csv --gt labels_private/win1.csv` and for `semantics` and
`diagnose` with `--gt`. Hand labels VALIDATE the system; they never go into a submission
(RULES.md section 1). They stay in labels_private/, which is gitignored.

Labeller protocol (PLAN.md section 7): first label the public window blind, write it out, and score
it as if it were a prediction -- `python -m possession.harness labels_private/public_<name>.csv` --
which measures how close the labeller is to the ground truth's conventions.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import ROOT  # noqa: F401  (puts the repository root on sys.path)
from . import log as plog
from .clock import clock_to_frame
from .config import DEFAULT

log = plog.get(__name__)
LABELS = ("H", "A", "D")


def parse_time(token, half_token=None):
    """A video frame from '12345', '00:20:31.40' (video time) or ('1H', '12:05') (match clock)."""
    if half_token is not None:
        half = half_token.upper().rstrip("H")
        if half not in ("1", "2"):
            raise ValueError("match clock needs 1H or 2H before it, got {!r}".format(half_token))
        return clock_to_frame(token, int(half))
    if token.isdigit():
        return int(token)
    parts = token.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(round((int(h) * 3600 + int(m) * 60 + float(s)) * DEFAULT.fps))
    raise ValueError("cannot read a time from {!r}: use a frame, hh:mm:ss.ss, or 1H/2H mm:ss".format(token))


def parse(text):
    """((first, last) window, [(frame, label)]) from the change-point text."""
    window, points = None, []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        try:
            if tokens[0].lower() == "window":
                window = (int(tokens[1]), int(tokens[2]))
                continue
            label = tokens[-1].upper()
            if label not in LABELS:
                raise ValueError("label must be H, A or D, got {!r}".format(tokens[-1]))
            if len(tokens) == 3:
                frame = parse_time(tokens[1], tokens[0])
            elif len(tokens) == 2:
                frame = parse_time(tokens[0])
            else:
                raise ValueError("expected '<time> <label>'")
        except (ValueError, IndexError) as exc:
            raise ValueError("line {}: {} -- {!r}".format(number, exc, raw.strip())) from None
        points.append((frame, label))
    if window is None:
        raise ValueError("no 'window <first> <last>' line")
    return window, points


def build(window, points, blocks, n_frames):
    """(n_frames,) labels: the change points inside the window, X everywhere else."""
    first, last = window
    if not any(a <= first and last < b for _, a, b in blocks):
        raise ValueError("window {}-{} is not inside one graded half {}".format(
            first, last, [(a, b - 1) for _, a, b in blocks]))
    if not points or points[0][0] != first:
        raise ValueError("the first change point must be the window's first frame {}".format(first))
    frames = [f for f, _ in points]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("change points must be strictly increasing")
    if frames[-1] > last:
        raise ValueError("change point {} is after the window's last frame {}".format(frames[-1], last))
    labels = np.full(n_frames, "X", dtype="<U1")
    ends = frames[1:] + [last + 1]
    for (f, lab), end in zip(points, ends):
        labels[f:end] = lab
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("changes", help="the change-point text file")
    ap.add_argument("--out", default=None, help="defaults to the same name with .csv")
    args = ap.parse_args(argv)
    plog.setup("labels")

    from games import GAMES
    from .ingest import graded_blocks
    from .policy import metrics, validate, write_csv

    with open(args.changes, encoding="utf-8") as handle:
        window, points = parse(handle.read())
    n = GAMES["BAR-ATH"]["n_video_frames"]
    labels = build(window, points, graded_blocks(), n)
    reason = validate(labels, n)
    if reason:
        raise ValueError(reason)
    out = args.out or os.path.splitext(args.changes)[0] + ".csv"
    write_csv(out, labels)
    pm = metrics()
    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    inside = labels[window[0]:window[1] + 1]
    log.info("window %d-%d (%.1f min): %d change points, %d possessions, D on %.1f%% of frames",
             window[0], window[1], len(inside) / DEFAULT.fps / 60.0, len(points), len(spells),
             100.0 * float((inside == "D").mean()))
    log.info("wrote %s -- use it as a ground truth: python -m possession.harness <prediction>.csv --gt %s",
             plog.rel(out), plog.rel(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
