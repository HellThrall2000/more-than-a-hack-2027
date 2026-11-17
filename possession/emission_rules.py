"""L4 version 1 -- rule-based per-frame evidence for H / A / D (PLAN.md 5.5).

Soft proximity and control give each team a logit, stoppage evidence gives D one, and Smart Tagging
adds its calibrated prior. A missing or uncertain ball produces *no* evidence for either team
(both logits fall to zero together), never evidence for a change: only a team's own proximity can
push its logit up. The weights are untuned first guesses; version 2 (learned) is not implemented.
"""

from __future__ import annotations

import numpy as np

from .ball.gaps import SRC_UNKNOWN
from .config import Config
from .features import FeatureSet
from .util import keep_long_runs

H, A, D = 0, 1, 2


def _logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def calm_evidence(fs: FeatureSet, cfg: Config):
    """(N,) in [0, 1]: the players say play has stopped (calm.py), for at least calm_min_frames,
    while nobody controls the ball and it is not travelling fast. Needs no ball at all."""
    calm = np.nan_to_num(fs.col("calm"))
    sustained = keep_long_runs(calm > 0.5, cfg.calm_min_frames)
    nobody = (fs.col("ctrl_h") == 0) & (fs.col("ctrl_a") == 0)
    not_fast = np.nan_to_num(fs.col("ball_speed"), nan=0.0) < 8.0
    return np.where(sustained & nobody & not_fast, calm, 0.0)


def stoppage_evidence(fs: FeatureSet, tagf, cfg: Config):
    """(N,) in [0, 1.5]: how strongly the frame says "play is stopped"."""
    still_walk = (fs.col("stationary") > 0) & (np.nan_to_num(fs.col("walk_frac")) > 0.6)
    ev = (fs.col("oob") + 1.0 * still_walk + cfg.emit_calm_weight * calm_evidence(fs, cfg)
          + 0.5 * tagf["setpiece"] * (fs.col("stationary") > 0) + 0.5 * tagf["goal"])
    return np.clip(ev, 0.0, 1.5), still_walk


def emit(fs: FeatureSet, tagf, cfg: Config, shape=None):
    """logp (N, 3): log P(H), log P(A), log P(D) per frame, rows sum to one in probability.

    `shape` (N,) is the player-shape team logit (shape.py), added only where no control episode
    speaks, so a controlled ball always decides."""
    sigma = np.nan_to_num(fs.col("sigma"), nan=cfg.emit_sigma_max)
    conf = np.clip(1.0 - sigma / cfg.emit_sigma_max, 0.0, 1.0)

    dead_ev, still_walk = stoppage_evidence(fs, tagf, cfg)
    alive_gate = 1.0 - 0.8 * still_walk      # a held-still ball with walking players is not play

    logits = np.zeros((len(conf), 3), dtype=np.float64)
    for team, tag in ((H, "h"), (A, "a")):
        d = np.nan_to_num(fs.col("d_" + tag), nan=1e3)
        rv = np.nan_to_num(fs.col("rv_" + tag), nan=cfg.emit_sigma_v)
        prox = np.exp(-d ** 2 / (2 * cfg.emit_sigma_d ** 2)) * np.exp(-rv ** 2 / (2 * cfg.emit_sigma_v ** 2))
        ctrl = fs.col("ctrl_" + tag) + 0.5 * fs.col("carrier_" + tag)
        logits[:, team] = alive_gate * (cfg.emit_kappa_prox * conf * prox + cfg.emit_kappa_ctrl * ctrl)
        logits[:, team] += cfg.emit_tag_weight * tagf["poss"][:, team]
    if shape is not None and cfg.emit_shape_weight > 0:
        # Only where the ball layer knows nothing: no control episode and the ball neither seen nor
        # reconstructed (src unknown / none). A carried or in-flight ball already names its team.
        nobody = (fs.col("ctrl_h") == 0) & (fs.col("ctrl_a") == 0) & (fs.col("src") >= SRC_UNKNOWN)
        # Moves belief between H and A only: the alive total log(e^H + e^A) is kept, so the players'
        # shape can never make a dead ball look alive.
        half_logit = 0.5 * cfg.emit_shape_weight * np.where(nobody, shape, 0.0)
        alive = _logsumexp(logits[:, :2])
        logits[:, H] += half_logit
        logits[:, A] -= half_logit
        logits[:, :2] += (alive - _logsumexp(logits[:, :2]))[:, None]
    logits[:, D] = cfg.emit_kappa_dead * dead_ev - cfg.emit_dead_bias
    return (logits - _logsumexp(logits)[:, None]).astype(np.float32)
