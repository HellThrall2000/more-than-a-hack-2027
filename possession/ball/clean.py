"""L2 step 1 -- reject physically impossible ball observations (PLAN.md 5.3.1)."""

from __future__ import annotations

import numpy as np

from ..config import Config


def clean(ball_xy, cfg: Config):
    """Return a copy of ball_xy with impossible detections set to NaN.

    An observation is rejected when it is further from the last accepted one than the speed cap
    allows (plus a slack for position noise). If `ball_reject_reset` consecutive observations are
    rejected but agree with each other, the *anchor* may have been the outlier, so they are restored
    and the anchor moves on -- otherwise one bad detection would delete the rest of the match. The
    chain of observations accepted before the jump is dropped too if the new run outweighs it (a
    short chain is the outlier); a long chain is kept, since then the ball itself jumped.
    """
    out = ball_xy.astype(np.float64).copy()
    orig = out.copy()
    obs = np.flatnonzero(~np.isnan(out[:, 0]))
    cap = cfg.ball_max_speed / cfg.fps
    last_f, last_p = None, None
    rejected = []
    chain = []
    for f in obs:
        p = orig[f]
        if last_f is None:
            last_f, last_p = f, p
            chain.append(f)
            continue
        if np.hypot(p[0] - last_p[0], p[1] - last_p[1]) <= cap * (f - last_f) + cfg.ball_jump_slack:
            last_f, last_p = f, p
            chain.append(f)
            rejected = []
            continue
        out[f] = np.nan
        rejected.append(f)
        if len(rejected) >= cfg.ball_reject_reset:
            pts = orig[rejected]
            steps = np.hypot(*(pts[1:] - pts[:-1]).T)
            gaps = np.diff(rejected)
            if np.all(steps <= cap * gaps + cfg.ball_jump_slack):
                out[rejected] = pts
                if len(chain) < len(rejected):
                    out[chain] = np.nan
                chain = list(rejected)
                last_f, last_p = rejected[-1], orig[rejected[-1]]
                rejected = []
            else:
                rejected = rejected[1:]
    return out.astype(np.float32)
