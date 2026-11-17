"""'Play has stopped', read from the players alone -- no ball, no labels (SITUATION.md S1, S2).

During a stoppage both teams walk; during play they run. The ball is usually missing exactly then
(out of view, in someone's hands), so without this the dead ball has no evidence at all.

The threshold between "walking" and "playing" is not set by hand. Each half's collective speed
(mean over all tracked players, smoothed over about 2 s, on a log scale) is fitted with a two-
component Gaussian mixture by EM, from fixed quantile starting points, so the result is
deterministic and adapts to the match. The posterior of the slow component is the calm
probability. No ground truth of any kind is involved.
"""

from __future__ import annotations

import numpy as np

from . import log as plog
from .config import Config
from .util import rolling_nanmean

log = plog.get(__name__)
EM_ITERATIONS = 60
MIN_SEPARATION = 1.0     # component means at least this many pooled standard deviations apart


def _normal_logpdf(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def fit_two_states(x):
    """EM for a 1-D two-component Gaussian mixture. Returns (means, variances, weights), slow first."""
    mu = np.percentile(x, [15, 85]).astype(np.float64)
    var = np.full(2, max(np.var(x) / 4.0, 1e-4))
    w = np.array([0.5, 0.5])
    for _ in range(EM_ITERATIONS):
        lp = np.stack([np.log(w[k]) + _normal_logpdf(x, mu[k], var[k]) for k in range(2)], axis=1)
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-9
        w = nk / nk.sum()
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = np.maximum((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk, 1e-4)
    order = np.argsort(mu)
    return mu[order], var[order], w[order]


def calm_probability(mean_speed, blocks, cfg: Config):
    """(N,) in [0, 1]: probability that play has stopped, from the players' collective speed.

    `mean_speed` is (N, 2), per team. Outside the graded blocks, and wherever the mixture finds no
    two clearly separated states, the probability is 0 -- no evidence, never a guess.
    """
    n = len(mean_speed)
    out = np.zeros(n, dtype=np.float32)
    with np.errstate(all="ignore"):
        both = np.nanmean(mean_speed, axis=1)
    smooth = rolling_nanmean(both, cfg.calm_smooth)
    for half, a, b in blocks:
        x = np.log(np.asarray(smooth[a:b], dtype=np.float64) + 0.1)
        ok = np.isfinite(x)
        if ok.sum() < 10 * cfg.fps:
            continue
        mu, var, w = fit_two_states(x[ok])
        separation = (mu[1] - mu[0]) / np.sqrt(var.mean())
        log.info("  calm, half %d: walking %.2f m/s vs playing %.2f m/s, %.0f%% of frames walking, "
                 "separation %.1f sd", half, np.exp(mu[0]) - 0.1, np.exp(mu[1]) - 0.1, 100 * w[0], separation)
        if separation < MIN_SEPARATION:
            log.warning("  calm, half %d: the two states overlap; no stoppage evidence from it", half)
            continue
        lp = np.stack([np.log(w[k]) + _normal_logpdf(x[ok], mu[k], var[k]) for k in range(2)], axis=1)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        seg = np.zeros(b - a, dtype=np.float32)
        seg[ok] = p[:, 0] / p.sum(axis=1)
        out[a:b] = seg
    return out
