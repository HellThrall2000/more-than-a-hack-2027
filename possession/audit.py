"""The first thing to run on the real bundle (PLAN.md 5.1 "Audit report").

    python -m possession.audit [--data-dir DIR]

Prints and saves AGGREGATE statistics only -- counts, percentiles, yes/no checks -- never
frame-level data, so the output can be shared in a chat or pasted into the technical document
without redistributing the match data. The questions it answers decide the design:

  1. does the ATD metadata declare ball channels beyond x/y/speed (height, state, owner)?
  2. how long are the ball gaps, and what do they look like?
  3. slot lifetimes and objects per frame
  4. does the tactical camera pan, and does the ball sit near the image centre?
  5. do the alignment and time-base checks in starter/ pass?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import ROOT
from . import log as plog
from .config import Config, DEFAULT
from .util import runs

log = plog.get(__name__)


def graded_mask(blocks, n):
    m = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        m[a:b] = True
    return m


def audit_arrays(inputs, cfg: Config = DEFAULT):
    """Statistics computable from the arrays alone (also used on the synthetic fixture)."""
    tr = inputs.tracking
    n = inputs.n_frames
    graded = graded_mask(inputs.blocks, n)
    seen = ~np.isnan(tr.ball_xy[:, 0])
    out = {"n_frames": n, "n_graded": int(graded.sum()),
           "ball_visible_all": float(seen.mean()), "ball_visible_graded": float(seen[graded].mean()),
           "ball_extra_channels": int(tr.ball_extra.shape[1])}
    if tr.ball_extra.shape[1]:
        out["ball_extra_nonnan"] = [float((~np.isnan(tr.ball_extra[:, i])).mean())
                                    for i in range(tr.ball_extra.shape[1])]

    gaps = []
    for _, a, b in inputs.blocks:
        g0, g1 = runs(~seen[a:b])
        gaps.extend((g1 - g0).tolist())
    gaps = np.array(gaps)
    if len(gaps):
        edges = [1, 5, 9, 26, 76, 251, 10 ** 9]
        hist = np.histogram(gaps, bins=edges)[0]
        out["gap_count"] = int(len(gaps))
        out["gap_percentiles_frames"] = {str(p): float(np.percentile(gaps, p)) for p in (50, 75, 90, 99)}
        out["gap_frames_share_by_bucket"] = {
            "1-4": 0, "5-8": 0, "9-25": 0, "26-75": 0, "76-250": 0, "251+": 0}
        totals = [float(gaps[(gaps >= lo) & (gaps < hi)].sum()) for lo, hi in zip(edges[:-1], edges[1:])]
        for key, tot in zip(out["gap_frames_share_by_bucket"], totals):
            out["gap_frames_share_by_bucket"][key] = tot / max(1.0, sum(totals))
        out["gap_count_by_bucket"] = {k: int(v) for k, v in zip(out["gap_frames_share_by_bucket"], hist)}

    present = ~np.isnan(tr.players_xy[:, :, 0])
    out["objects_per_graded_frame_mean"] = float(present[graded].sum(axis=1).mean())
    out["frames_over_22_objects"] = float((present[graded].sum(axis=1) > 22).mean())
    lifetimes = []
    for s in range(present.shape[1]):
        a, b = runs(present[:, s] & graded)
        lifetimes.extend((b - a).tolist())
    if lifetimes:
        out["slot_lifetime_s"] = {str(p): float(np.percentile(lifetimes, p) / cfg.fps)
                                  for p in (10, 50, 90)}
        out["n_slot_tracklets"] = len(lifetimes)
    return out


def audit_metadata(meta_path):
    """Which channel / ball-related declarations does the ATD metadata XML make?"""
    root = ET.parse(meta_path).getroot()
    tags = {}
    ball_mentions = set()
    for el in root.iter():
        name = el.tag.split("}")[-1]
        tags[name] = tags.get(name, 0) + 1
        blob = " ".join([name] + ["%s=%s" % kv for kv in el.attrib.items()] + [(el.text or "").strip()])
        if "ball" in blob.lower():
            ball_mentions.add(blob.strip()[:120])
    return {"element_counts": tags, "ball_mentions": sorted(ball_mentions)[:20]}


def audit_camera(ball_xy, homography, valid, blocks, cfg: Config, step=250):
    """Does the tactical camera pan? Where does the image centre land on the pitch?"""
    from starter.load_homography import unproject
    pts, dists = [], []
    for _, a, b in blocks:
        for f in range(a, b, step):
            if not valid[f]:
                continue
            c = np.array(unproject(homography[f], 960.0, 540.0)) * [cfg.pitch_length, cfg.pitch_width]
            pts.append(c)
            if not np.isnan(ball_xy[f, 0]):
                dists.append(float(np.hypot(*(c - ball_xy[f]))))
    pts = np.array(pts)
    return {"image_centre_on_pitch_std_m": [float(pts[:, 0].std()), float(pts[:, 1].std())] if len(pts) else None,
            "ball_to_image_centre_m": {str(p): float(np.percentile(dists, p)) for p in (50, 90)}
            if dists else None,
            "n_sampled": int(len(pts))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "cache", "audit.json"))
    args = ap.parse_args(argv)
    plog.setup("audit")

    from games import require
    from .ingest import load_match
    from starter import load_atd, load_homography, load_smart_tagging

    cfg = DEFAULT
    with plog.stage("load the match"):
        inputs = load_match(cfg, args.data_dir)
    with plog.stage("array statistics"):
        report = {"arrays": audit_arrays(inputs, cfg)}
        report["metadata"] = audit_metadata(require(["atd_meta"], data_dir=args.data_dir)[0])
    with plog.stage("alignment checks (ATD, homography, Smart Tagging time base)"):
        report["alignment_atd"] = load_atd.check_alignment_raw(data_dir=args.data_dir)
        report["alignment_homography"] = {k: bool(v) if isinstance(v, (bool, np.bool_)) else v
                                          for k, v in load_homography.check_alignment(data_dir=args.data_dir).items()}
        report["time_base_tags"] = load_smart_tagging.check_time_base(inputs.tags)
    if inputs.homography is not None:
        with plog.stage("camera"):
            report["camera"] = audit_camera(inputs.tracking.ball_xy, inputs.homography,
                                            inputs.homography_valid, inputs.blocks, cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    log.info("audit report, also saved to %s:\n%s", plog.rel(args.out),
             json.dumps(report, indent=2, default=str))
    ok = report["alignment_atd"]["agrees"] and report["time_base_tags"]["agrees"]
    if ok:
        log.info("ALIGNMENT OK")
    else:
        log.error("ALIGNMENT MISMATCH -- stop and report it, do not patch around it")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
