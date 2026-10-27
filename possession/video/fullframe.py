"""Full-frame ball detection with a public, pretrained football ball detector -- no training.

    python -m possession.video.fullframe --evaluate     # self-check on hidden visible balls (aggregates)
    python -m possession.video.fullframe                # detect on every missing-ball graded frame
    python -m possession.run --video-ball cache/video_ball_fullframe.npz

The model is Roboflow's `football-ball-detection` (YOLOv8, from github.com/roboflow/sports, MIT
repository), downloaded once into models/ (gitignored) from the public link its own setup script uses.
Its SHA-256 is logged so a rerun can confirm the same weights.

Why not the detector in train.py: that one searches only a small crop around the gap filler's guess
and skips frames whose guess is uncertain by more than 6 m. In the long gaps the guess is about 13 m
off, so it searched almost nothing (gate coverage 0.02%). This one searches the whole frame on the GPU.

The self-check hides nothing from the model -- it simply runs on frames where the tracking DOES see
the ball and measures how often the detection lands within 2 m of it. That picks the confidence
threshold (precision >= 0.9) without any possession label. Detections become ordinary ball
observations: the ball layer's speed gate still rejects any that jump impossibly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading

import numpy as np

from .. import ROOT
from .. import log as plog
from ..config import Config, DEFAULT
from .frames import gray_frames
from .geometry import pitch_to_px, px_to_pitch

log = plog.bind(__name__)

MODEL_PATH = os.path.join(ROOT, "models", "football-ball-detection.pt")
GDRIVE_ID = "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"     # roboflow/sports examples/soccer/setup.sh
GATE_FILE = os.path.join(ROOT, "cache", "video_gate_fullframe.json")
OUT_FILE = os.path.join(ROOT, "cache", "video_ball_fullframe.npz")
TOPK = 5                    # candidates kept per frame for the filters and the temporal linker
PITCH_MARGIN_M = 2.0        # a match ball more than this far outside the lines is a spare ball
STATIC_CELL_M = 0.5         # grid for finding objects that never move
STATIC_SHARE = 0.01         # a cell holding detections on more than this share of frames is static ...
STATIC_MIN_FRAMES = 250     # ... and on at least 10 s worth of frames (a ball at rest for a restart is shorter)
MIN_CONF = 0.05             # keep weak candidates; the threshold is chosen by the self-check
HIT_M = 2.0                 # a detection within this of the tracking's ball counts as right


def ensure_weights(path=MODEL_PATH):
    if not os.path.isfile(path):
        import gdown
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log.info("downloading weights", source="roboflow/sports football-ball-detection", to=plog.rel(path))
        tmp = path + ".part"
        if gdown.download(id=GDRIVE_ID, output=tmp, quiet=True) is None:
            raise RuntimeError("weights download failed; get football-ball-detection.pt from roboflow/sports "
                               "(examples/soccer/setup.sh) and put it at {}".format(path))
        os.replace(tmp, path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    log.info("weights", file=plog.rel(path), sha256=digest[:16])
    return path


def load_model(device):
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)     # one deprecation warning per batch otherwise
    from ultralytics import YOLO
    model = YOLO(ensure_weights())
    model.to(device)
    return model


def _batches(video, frames, expected, size):
    """Background decoding thread -> lists of (frame index, BGR image), `size` at a time."""
    q = queue.Queue(maxsize=4)
    err = []

    def produce():
        try:
            batch = []
            for idx, img in gray_frames(video, frames, expected, "decoding for ball detection", colour=True):
                batch.append((idx, img))
                if len(batch) == size:
                    q.put(batch)
                    batch = []
            if batch:
                q.put(batch)
        except BaseException as exc:            # noqa: BLE001 -- re-raised in the consumer
            err.append(exc)
        finally:
            q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    if err:
        raise err[0]


def detect_px(video, frames, model, device="cuda", imgsz=1280, batch=8, expected=None):
    """(cand_px (n, TOPK, 2), cand_conf (n, TOPK)) in image pixels for the sorted `frames`."""
    frames = np.asarray(frames, dtype=np.int64)
    pos = {int(f): i for i, f in enumerate(frames)}
    px = np.full((len(frames), TOPK, 2), np.nan, dtype=np.float32)
    conf = np.zeros((len(frames), TOPK), dtype=np.float32)
    half = str(device).startswith("cuda")
    for items in _batches(video, frames, expected, batch):
        res = model.predict([img for _, img in items], imgsz=imgsz, conf=MIN_CONF, half=half, device=device,
                            verbose=False, max_det=TOPK)
        for (idx, _), r in zip(items, res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            c = b.conf.float().cpu().numpy()
            xy = b.xywh[:, :2].float().cpu().numpy()
            order = np.argsort(-c, kind="stable")[:TOPK]
            i = pos[int(idx)]
            px[i, :len(order)], conf[i, :len(order)] = xy[order], c[order]
    return px, conf


def to_pitch(px, frames, homography, cfg: Config):
    """Candidate pixels (n, K, 2) -> pitch metres (n, K, 2)."""
    n, k, _ = px.shape
    hs = np.repeat(homography[frames], k, axis=0)
    flat = px.reshape(-1, 2).astype(np.float64)
    ok = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan)
    if ok.any():
        out[ok] = px_to_pitch(hs[ok], flat[ok], cfg)
    return out.reshape(n, k, 2)


def plausible(cand_xy, cand_conf, cfg: Config):
    """Candidate confidences with the impossible ones zeroed: off the pitch (spare balls along the
    touchlines, ball boys) or at a spot that holds a detection on many frames (painted penalty and
    centre spots, a ball left by the dugout). The match ball moves; those do not.
    Returns (conf (n, K), counts by reason)."""
    conf = cand_conf.copy()
    x, y = cand_xy[..., 0], cand_xy[..., 1]
    m = PITCH_MARGIN_M
    with np.errstate(invalid="ignore"):
        off = ~((x >= -m) & (x <= cfg.pitch_length + m) & (y >= -m) & (y <= cfg.pitch_width + m))
    off &= conf > 0
    conf[off] = 0.0
    live = conf > 0
    cell = np.where(live[..., None], np.floor(cand_xy / STATIC_CELL_M), np.nan)
    static = np.zeros_like(live)
    if live.any():
        keys = cell[live].astype(np.int64)
        flat = keys[:, 0] * 100000 + keys[:, 1]
        uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
        # a detection plus its 8 neighbours: an object sitting on a cell edge still counts once
        pos = {int(k): i for i, k in enumerate(uniq)}
        neigh = np.array([sum(cnt[pos[k + dx * 100000 + dy]] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                              if (k + dx * 100000 + dy) in pos) for k in uniq.tolist()])
        heavy = neigh > max(STATIC_SHARE * len(cand_xy), STATIC_MIN_FRAMES)
        static[live] = heavy[inv]
    conf[static] = 0.0
    return conf, {"off_pitch": int(off.sum()), "static": int(static.sum())}


def tracklets(frames, cand_xy, cand_conf, thr, cfg: Config, max_gap=5, min_len=8):
    """One position per frame, taken only from steady chains of detections.

    Candidates at or above `thr` are chained frame to frame when the jump is one a ball can make
    (cfg.ball_max_speed plus slack) and the frames are at most `max_gap` apart. A chain shorter than
    `min_len` detections -- a boot, a head, a flicker -- is dropped. Where several chains survive in a
    frame, the longest wins. Returns (xy (n, 2), conf (n,)); NaN / 0 where nothing survives.
    """
    n, k = cand_conf.shape
    chain = np.full((n, k), -1, dtype=np.int64)
    lengths = []
    active = []                                  # [chain id, last frame, last position]
    for i in range(n):
        active = [c for c in active if frames[i] - c[1] <= max_gap]
        idx = np.flatnonzero(cand_conf[i] >= thr)
        used = set()
        for j in idx[np.argsort(-cand_conf[i, idx], kind="stable")]:
            p = cand_xy[i, j]
            best, best_d = None, np.inf
            for c in active:
                if c[0] in used:
                    continue
                reach = cfg.ball_max_speed * (frames[i] - c[1]) / cfg.fps + cfg.ball_jump_slack
                d = float(np.hypot(*(p - c[2])))
                if d <= reach and d < best_d:
                    best, best_d = c, d
            if best is None:
                best = [len(lengths), frames[i], p]
                lengths.append(0)
                active.append(best)
            used.add(best[0])
            best[1], best[2] = frames[i], p
            lengths[best[0]] += 1
            chain[i, j] = best[0]
    lengths = np.asarray(lengths + [0])
    size = np.where(chain >= 0, lengths[chain], 0)
    size[size < min_len] = 0
    xy = np.full((n, 2), np.nan)
    conf = np.zeros(n)
    for i in np.flatnonzero(size.max(axis=1) > 0):
        j = int(np.lexsort((-cand_conf[i], -size[i]))[0])
        xy[i], conf[i] = cand_xy[i, j], cand_conf[i, j]
    return xy, conf


def link(frames, cand_xy, cand_conf, thr, cfg: Config, max_gap=10):
    """One position per frame: the candidate nearest the previous accepted one (within what the ball
    can travel), else the most confident. Returns (xy (n, 2), conf (n,)); NaN / 0 where none passes."""
    n = len(frames)
    xy = np.full((n, 2), np.nan)
    conf = np.zeros(n)
    last_f, last_xy = None, None
    for i in range(n):
        ok = cand_conf[i] >= thr
        if not ok.any():
            continue
        cands, cc = cand_xy[i][ok], cand_conf[i][ok]
        j = 0
        if last_f is not None and frames[i] - last_f <= max_gap:
            d = np.hypot(*(cands - last_xy).T)
            reach = cfg.ball_max_speed * (frames[i] - last_f) / cfg.fps + cfg.ball_jump_slack
            near = d <= reach
            if near.any():
                j = int(np.flatnonzero(near)[np.argmin(d[near])])
        xy[i], conf[i] = cands[j], cc[j]
        last_f, last_xy = frames[i], cands[j]
    return xy, conf


def self_check(cand_xy, cand_conf, truth_xy):
    """Coverage and precision of the top candidate against the tracking's own ball, per threshold."""
    err = np.hypot(*(cand_xy[:, 0] - truth_xy).T)
    rows = []
    for thr in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7):
        found = cand_conf[:, 0] >= thr
        prec = float((err[found] <= HIT_M).mean()) if found.any() else None
        rows.append({"thr": thr, "coverage": round(float(found.mean()), 4),
                     "precision": None if prec is None else round(prec, 4)})
    good = [r for r in rows if r["precision"] is not None and r["precision"] >= 0.9]
    return rows, (min(good, key=lambda r: r["thr"])["thr"] if good else None)


def explain_errors(px, conf, frames, truth, homography, cfg: Config, thr=0.3):
    """Aggregate-only breakdown of why detections miss the tracking's ball.

    Compares in PIXELS (detection vs the tracking ball projected into the image) and in metres, scans
    a +/- 5 frame offset, and splits by where the ball is. A pure shift or scale shows up as a steady
    median offset or a ratio away from 1; a frame offset as a better score at a non-zero lag.
    """
    ok = conf[:, 0] >= thr
    f = frames[ok]
    det_px = px[ok, 0].astype(np.float64)
    proj = pitch_to_px(homography[f], truth[f], cfg)
    d = det_px - proj
    dist_px = np.hypot(*d.T)
    det_m = px_to_pitch(homography[f], det_px, cfg)
    err_m = np.hypot(*(det_m - truth[f]).T)
    log.info("errors in pixels", n=int(ok.sum()), median_px=round(float(np.median(dist_px)), 1),
             p90_px=round(float(np.percentile(dist_px, 90)), 1),
             median_dx=round(float(np.median(d[:, 0])), 1), median_dy=round(float(np.median(d[:, 1])), 1),
             within_10px=round(float((dist_px <= 10).mean()), 3), within_30px=round(float((dist_px <= 30).mean()), 3))
    log.info("errors in metres", median_m=round(float(np.median(err_m)), 2),
             p90_m=round(float(np.percentile(err_m, 90)), 1),
             within_2m=round(float((err_m <= 2).mean()), 3), within_5m=round(float((err_m <= 5).mean()), 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        cx, cy = 960.0, 540.0
        rx = np.median((det_px[:, 0] - cx) / (proj[:, 0] - cx))
        ry = np.median((det_px[:, 1] - cy) / (proj[:, 1] - cy))
    log.info("scale check (1.0 = no scale error)", ratio_x=round(float(rx), 3), ratio_y=round(float(ry), 3),
             proj_px_range_x=[round(float(np.percentile(proj[:, 0], q)), 0) for q in (1, 99)],
             proj_px_range_y=[round(float(np.percentile(proj[:, 1], q)), 0) for q in (1, 99)],
             det_px_range_x=[round(float(np.percentile(det_px[:, 0], q)), 0) for q in (1, 99)],
             det_px_range_y=[round(float(np.percentile(det_px[:, 1], q)), 0) for q in (1, 99)])
    lags = {}
    for lag in range(-5, 6):
        g = np.clip(f + lag, 0, len(truth) - 1)
        good = ~np.isnan(truth[g, 0])
        e = np.hypot(*(det_m[good] - truth[g][good]).T)
        lags[str(lag)] = round(float((e <= 2).mean()), 3)
    log.info("within 2 m if the tracking is shifted by k frames", **lags)
    y = truth[f, 1]
    for name, sel in (("near touchline third", y > 2 * cfg.pitch_width / 3),
                      ("middle third", (y >= cfg.pitch_width / 3) & (y <= 2 * cfg.pitch_width / 3)),
                      ("far touchline third", y < cfg.pitch_width / 3)):
        if sel.any():
            log.info("by pitch width", part=name, n=int(sel.sum()), within_2m=round(float((err_m[sel] <= 2).mean()), 3),
                     median_px=round(float(np.median(dist_px[sel])), 1))
    near = np.hypot(*(to_pitch(px[ok], f, homography, cfg) - truth[f][:, None, :]).transpose(2, 0, 1))
    log.info("best of the remaining candidates", within_2m=round(float((np.nanmin(near, axis=1) <= 2).mean()), 3))


def compare(path_a, path_b):
    """Do two detection runs agree? Prints counts only. Exit 0 if identical, 4 if not."""
    a, b = np.load(path_a), np.load(path_b)
    used_a = a["conf"] >= a["thr"]
    used_b = b["conf"] >= b["thr"]
    both = used_a & used_b
    moved = np.hypot(*(a["xy"][both] - b["xy"][both]).T) if both.any() else np.zeros(0)
    same = (np.array_equal(a["xy"], b["xy"], equal_nan=True) and np.array_equal(a["conf"], b["conf"])
            and float(a["thr"]) == float(b["thr"]))
    log.info("detection runs compared", identical=bool(same), used_a=int(used_a.sum()), used_b=int(used_b.sum()),
             used_by_only_one=int((used_a ^ used_b).sum()), moved_over_1cm=int((moved > 0.01).sum()),
             max_move_m=round(float(moved.max()), 3) if moved.size else 0.0)
    return 0 if same else 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evaluate", action="store_true", help="self-check on frames where the tracking sees the ball")
    ap.add_argument("--n-check", type=int, default=3000)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="detect even if the self-check did not reach 0.9: use the threshold with the best "
                         "precision (an experiment -- compare the resulting CSV before trusting it)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=OUT_FILE, help="where the detections go (default %(default)s)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two detection files and print only whether they agree (aggregates)")
    args = ap.parse_args(argv)
    if args.compare:
        plog.setup("video_compare")
        return compare(*args.compare)
    plog.setup("video_fullframe_check" if args.evaluate else "video_fullframe")
    from games import asset_paths
    from ..ingest import load_match
    from ..ball import clean as _clean
    cfg = DEFAULT
    inputs = load_match(cfg, args.data_dir)
    video = asset_paths(data_dir=args.data_dir)["video"]
    try:
        model = load_model(args.device)
    except Exception as exc:                   # noqa: BLE001 -- say what to do, then stop
        log.error("detector unavailable", reason=str(exc))
        return 1
    truth = _clean.clean(inputs.tracking.ball_xy, cfg)
    graded = np.zeros(inputs.n_frames, dtype=bool)
    for _, a, b in inputs.blocks:
        graded[a:b] = True
    seen = graded & ~np.isnan(truth[:, 0])

    if args.evaluate:
        rng = np.random.RandomState(cfg.seed)
        win = 75                                     # 3 s windows, so the tracklet filter has context
        cover = np.convolve(seen.astype(float), np.ones(win), "valid") / win
        starts = np.flatnonzero(cover >= 0.9)
        starts = np.sort(rng.choice(starts, size=min(args.n_check // win, len(starts)), replace=False))
        frames = np.unique((starts[:, None] + np.arange(win)[None, :]).ravel())
        with plog.stage("self-check: detect on {} frames where the tracking sees the ball".format(len(frames))):
            px, cc = detect_px(video, frames, model, args.device, args.imgsz, args.batch, inputs.n_frames)
        cand = to_pitch(px, frames, inputs.homography, cfg)
        raw_rows, _ = self_check(cand, cc, truth[frames])
        log.info("before the filters", **{"precision@0.3": next(r["precision"] for r in raw_rows if r["thr"] == 0.3)})
        fc, removed = plausible(cand, cc, cfg)
        log.info("filters removed candidates", **removed)
        order = np.argsort(-fc, axis=1, kind="stable")
        cand, fc = np.take_along_axis(cand, order[..., None], axis=1), np.take_along_axis(fc, order, axis=1)
        px = np.take_along_axis(px, order[..., None], axis=1)
        has = seen[frames]
        rows, _ = self_check(cand[has], fc[has], truth[frames][has])
        for r in rows:
            log.info("self-check, single frames", **r)
        explain_errors(px[has], fc[has], frames[has], truth, inputs.homography, cfg)
        rows = []
        for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            xy_t, c_t = tracklets(frames, cand, fc, t, cfg)
            found = (c_t > 0) & has
            err = np.hypot(*(xy_t[found] - truth[frames][found]).T)
            rows.append({"thr": t, "coverage": round(float(found.sum() / has.sum()), 4),
                         "precision": round(float((err <= HIT_M).mean()), 4) if found.any() else None,
                         "within_5m": round(float((err <= 5).mean()), 4) if found.any() else None})
            log.info("self-check, steady tracks only", **rows[-1])
        good = [r for r in rows if r["precision"] is not None and r["precision"] >= 0.9]
        thr = min(good, key=lambda r: r["thr"])["thr"] if good else None
        result = {"n": int(len(frames)), "imgsz": args.imgsz, "rows": rows, "thr": thr}
        os.makedirs(os.path.dirname(GATE_FILE), exist_ok=True)
        with open(GATE_FILE, "w") as fh:
            json.dump(result, fh, indent=2)
        if thr is None:
            log.error("GATE NOT PASSED: no threshold reaches precision 0.9 -- do not use the detections")
            return 3
        log.info("GATE PASSED", threshold=thr, coverage=next(r["coverage"] for r in rows if r["thr"] == thr),
                 note="steady tracks only")
        return 0

    if not os.path.isfile(GATE_FILE):
        log.error("run the self-check first: python -m possession.video.fullframe --evaluate")
        return 1
    with open(GATE_FILE) as fh:
        gate = json.load(fh)
    thr = gate.get("thr")
    if thr is None and args.force and gate.get("rows"):
        best = max((r for r in gate["rows"] if r["precision"] is not None), key=lambda r: (r["precision"], -r["thr"]))
        thr = best["thr"]
        log.warning("FORCED: the self-check did not pass; using the best threshold anyway",
                    threshold=thr, precision=best["precision"], coverage=best["coverage"])
    if thr is None:
        log.error("the self-check did not pass; not detecting (add --force to try anyway)")
        return 3
    thr = float(thr)
    frames = np.flatnonzero(graded & np.isnan(truth[:, 0]))
    with plog.stage("detect on {} graded frames without a tracked ball".format(len(frames))):
        px, cc = detect_px(video, frames, model, args.device, args.imgsz, args.batch, inputs.n_frames)
    cand = to_pitch(px, frames, inputs.homography, cfg)
    fc, removed = plausible(cand, cc, cfg)
    log.info("filters removed candidates", **removed)
    xy_f, conf_f = tracklets(frames, cand, fc, thr, cfg)
    xy = np.full((inputs.n_frames, 2), np.nan, dtype=np.float32)
    conf = np.zeros(inputs.n_frames, dtype=np.float32)
    xy[frames] = np.round(xy_f, 2)
    conf[frames] = np.round(conf_f, 3)
    np.savez_compressed(args.out, xy=xy, conf=conf, thr=np.float32(thr))
    log.info("detections saved", file=plog.rel(args.out), recovered=int((conf_f >= thr).sum()),
             searched=len(frames), share=round(float((conf_f >= thr).mean()), 4) if len(frames) else 0.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
