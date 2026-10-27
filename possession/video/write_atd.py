"""Write our own ATD: the organisers' file, line for line, with missing ball fields filled from the video.

    python -m possession.video.write_atd                 # -> cache/atd_ours/tactical/ (gitignored)
    MTAH_DATA_DIR=cache/atd_ours python -m possession.run --out submissions/x.csv

Every line keeps its frame id and every player group byte for byte. Only a ball group that is
missing (starts with "N") AND has a video detection at or above the validated threshold is replaced,
by "x,y,speed" in the same normalised 0-1 coordinates and with the same number of decimals as the
original ball groups. The other assets (metadata, homography, Smart Tagging, video) are linked, not
copied. `verify` re-parses both files and checks that nothing but those ball groups changed.

Everything here is frame-level data derived from the match: it lives in cache/ and is never committed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .. import ROOT
from .. import log as plog
from ..ingest import FIELD_SEP, parse_atd

log = plog.bind(__name__)

OUT_ROOT = os.path.join(ROOT, "cache", "atd_ours")
DETECTIONS = os.path.join(ROOT, "cache", "video_ball_fullframe.npz")


def _ball_format(raw_path):
    """(number of fields, decimals per field) of the first present ball group: structure only."""
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) == 3 and parts[2] and parts[2][0] != "N":
                fields = parts[2].split(",")
                return len(fields), [len(f.split(".")[1]) if "." in f else 0 for f in fields]
    raise ValueError("{}: no ball observation to copy the format from".format(raw_path))


def _speeds(xy, fps):
    """Ball speed (m/s) per frame from the filled track, central difference; 0 where undefined."""
    v = np.zeros(len(xy))
    v[1:-1] = np.hypot(*(xy[2:] - xy[:-2]).T) * fps / 2.0
    return np.nan_to_num(v)


def write(raw_path, out_path, det_xy, det_conf, thr, pitch, fps, inside_m=None):
    """Copy `raw_path` to `out_path`, filling missing ball groups. Returns the number filled.

    With `inside_m`, only detections at least that far inside the touchlines and goal lines are used,
    so a video detection can never put the ball out of play on its own."""
    n_fields, decimals = _ball_format(raw_path)
    length, width = pitch
    use = np.isfinite(det_xy).all(axis=1) & (det_conf >= thr)
    if inside_m is not None:
        with np.errstate(invalid="ignore"):
            use &= ((det_xy[:, 0] >= inside_m) & (det_xy[:, 0] <= length - inside_m)
                    & (det_xy[:, 1] >= inside_m) & (det_xy[:, 1] <= width - inside_m))
    speed = _speeds(np.where(use[:, None], det_xy, np.nan), fps)
    filled = 0
    tmp = out_path + ".part"
    with open(raw_path) as src, open(tmp, "w", newline="") as dst:
        for line in src:
            end = line[len(line.rstrip("\r\n")):]
            parts = line[:len(line) - len(end)].split(FIELD_SEP)
            if len(parts) == 3 and (not parts[2] or parts[2][0] == "N"):
                frame = int(parts[0]) - 1
                if 0 <= frame < len(use) and use[frame]:
                    values = [det_xy[frame, 0] / length, det_xy[frame, 1] / width, speed[frame]]
                    values += [0.0] * max(0, n_fields - 3)
                    parts[2] = ",".join("{:.{}f}".format(v, d) for v, d in zip(values[:n_fields], decimals))
                    line = FIELD_SEP.join(parts) + end
                    filled += 1
            dst.write(line)
    os.replace(tmp, out_path)
    return filled


def verify(raw_path, out_path, n_frames, pitch):
    """Re-parse both files: players identical, original balls identical, only gaps filled."""
    a, b = parse_atd(raw_path, n_frames, pitch), parse_atd(out_path, n_frames, pitch)
    players_same = (np.array_equal(a.players_xy, b.players_xy, equal_nan=True)
                    and np.array_equal(a.players_speed, b.players_speed, equal_nan=True))
    had = ~np.isnan(a.ball_xy[:, 0])
    balls_same = np.array_equal(a.ball_xy[had], b.ball_xy[had])
    new = (~had) & ~np.isnan(b.ball_xy[:, 0])
    inside = np.isfinite(b.ball_xy[new]).all()
    return {"players_identical": bool(players_same), "original_balls_identical": bool(balls_same),
            "frames_filled": int(new.sum()), "ball_seen_before": round(float(had.mean()), 4),
            "ball_seen_after": round(float((~np.isnan(b.ball_xy[:, 0])).mean()), 4), "values_finite": bool(inside),
            "line_count_same": sum(1 for _ in open(raw_path)) == sum(1 for _ in open(out_path))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detections", default=DETECTIONS)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--inside-m", type=float, default=None,
                    help="use only detections at least this many metres inside the lines")
    args = ap.parse_args(argv)
    plog.setup("write_atd")
    from games import GAMES, asset_paths
    if not os.path.isfile(args.detections):
        log.error("no detections: run python -m possession.video.fullframe first", missing=plog.rel(args.detections))
        return 1
    z = np.load(args.detections)
    src = asset_paths(data_dir=args.data_dir)
    out = asset_paths(data_dir=args.out_root)
    os.makedirs(os.path.dirname(out["atd_raw"]), exist_ok=True)
    for key, path in src.items():
        if key != "atd_raw" and os.path.exists(path) and not os.path.lexists(out[key]):
            os.symlink(os.path.abspath(path), out[key])
    game = GAMES["BAR-ATH"]
    with plog.stage("write our ATD"):
        filled = write(src["atd_raw"], out["atd_raw"], z["xy"], z["conf"], float(z["thr"]), game["pitch"],
                       game["fps"], args.inside_m)
    log.info("ball groups filled from the video", count=filled, file=plog.rel(out["atd_raw"]))
    with plog.stage("verify against the original"):
        report = verify(src["atd_raw"], out["atd_raw"], game["n_video_frames"], game["pitch"])
    log.info("verification", **report)
    ok = report["players_identical"] and report["original_balls_identical"] and report["line_count_same"] \
        and report["values_finite"]
    log.info("OUR ATD IS FORMAT-COMPLIANT" if ok else "OUR ATD FAILED VERIFICATION", data_dir=plog.rel(args.out_root))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
