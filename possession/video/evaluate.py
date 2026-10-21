"""The go / no-go gate (PLAN.md section 11): can the detector recover hidden, previously seen balls?

Visible stretches of the ATD ball are hidden using the real gap-length distribution -- inside the
validation time blocks only, which the detector never trained on -- the ball layer fills them as it
normally would, and the detector then searches around that fill. Truth is the ATD's own position,
so this measures agreement with the tracker, not with reality; that is also why the tolerance is 2 m.
"""

from __future__ import annotations

import numpy as np

from ..ball import clean as _clean
from ..ball import smooth as _smooth
from ..ball.gaps import reconstruct
from ..ball.scorecard import BUCKETS, mask_windows
from ..config import Config
from ..util import runs
from . import VideoConfig
from .infer import detect


def evaluate(model, vcfg: VideoConfig, thr, video_path, ball_xy, players_xy, slot_team, homography,
             blocks, cfg: Config, n_windows=150, seed=0, hit_m=2.0, device="cpu", expected_frames=None):
    rng = np.random.RandomState(seed)
    truth = _clean.clean(ball_xy, cfg)
    visible = ~np.isnan(truth[:, 0])
    g0, g1 = runs(~visible)
    lengths = (g1 - g0)[(g1 - g0) >= BUCKETS[0][0]]
    seg = np.arange(len(truth)) // vcfg.val_segment_frames
    held_out = visible & ((seg % vcfg.val_every) == (vcfg.val_every - 1))

    windows = mask_windows(held_out, lengths, rng, n_windows)
    masked = truth.copy()
    wlen = np.zeros(len(truth), dtype=np.int32)
    for s, length in windows:
        masked[s:s + length] = np.nan
        wlen[s:s + length] = length
    pos, vel, observed, interp = _smooth.smooth_ball(masked, cfg)
    gaps = reconstruct(pos, vel, observed & ~np.isnan(pos[:, 0]), interp, players_xy, slot_team, blocks, cfg)

    frames = np.flatnonzero(wlen > 0)
    xy, conf = detect(video_path, frames, gaps.xy[frames], gaps.sigma[frames], homography, model, cfg, vcfg,
                      device, expected_frames)
    found = conf >= thr
    err_det = np.hypot(*(xy - truth[frames]).T)
    err_fill = np.hypot(*(gaps.xy[frames] - truth[frames]).T)
    err_both = np.where(found, err_det, err_fill)

    out = {"n_windows": len(windows), "n_frames": int(len(frames)), "threshold": float(thr), "by_length": {}}
    for lo, hi in BUCKETS:
        sel = (wlen[frames] >= lo) & (wlen[frames] <= hi)
        if not sel.any():
            continue
        d = found & sel
        out["by_length"]["%d-%s" % (lo, hi if hi < 10 ** 6 else "inf")] = {
            "frames": int(sel.sum()),
            "coverage": float(d.sum()) / sel.sum(),
            "precision": float((err_det[d] <= hit_m).mean()) if d.any() else None,
            "filler_median_m": float(np.nanmedian(err_fill[sel])),
            "filler_plus_video_median_m": float(np.nanmedian(err_both[sel]))}
    d = found
    out["overall"] = {"coverage": float(d.mean()), "precision": float((err_det[d] <= hit_m).mean()) if d.any() else None,
                      "filler_median_m": float(np.nanmedian(err_fill)),
                      "filler_plus_video_median_m": float(np.nanmedian(err_both))}
    prec = out["overall"]["precision"]
    out["gate"] = {"precision_required": vcfg.gate_precision, "met": bool(prec is not None and prec >= vcfg.gate_precision),
                   "worthwhile": bool(out["overall"]["coverage"] >= 0.2)}
    return out
