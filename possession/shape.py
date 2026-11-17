"""Which team has the ball, read from the players alone (no ball needed).

The ball is missing on 41% of graded frames, mostly in gaps longer than 10 s, and there the ball
layer has nothing to say about the team. The players still do: the team in possession pushes up and
spreads out, the other team drops and narrows, and both move with the play.

A small gradient-boosted tree model learns that mapping from this match's own frames where the ball
IS seen and a control episode says who has it (control alone agrees with the ground truth about 93%
of the time on the public window). No hand labels, no ground truth, no external data. It is fitted
in seconds on the CPU, deterministically, every time the pipeline runs -- the same way the Smart
Tagging weights are calibrated -- so there are no weights to ship.

`team_logit` returns log P(home) - log P(away) per frame; the emission adds it where no control
episode speaks (SITUATION.md S12 still decides wherever the ball is controlled).
"""

from __future__ import annotations

import numpy as np

from . import log as plog
from .config import Config
from .util import rolling_nanmean

log = plog.get(__name__)

BLOCK = 3000            # frames per cross-validation block (2 minutes): neighbours never leak
HOLD_EVERY = 5          # every 5th block is held out for the accuracy check


def shape_features(players_xy, slot_team, attack, blocks, cfg: Config):
    """(N, F) float32 features in each team's own attacking frame, and their names."""
    n = len(players_xy)
    length, width = cfg.pitch_length, cfg.pitch_width
    x, y = players_xy[:, :, 0], players_xy[:, :, 1]
    cols = {}
    sign = np.zeros((n, 2), dtype=np.float32)          # +1: this team attacks towards +x in this frame
    for half, a, b in blocks:
        sign[a:b] = attack[half]
    with np.errstate(all="ignore"):
        for t, tag in ((0, "h"), (1, "a")):
            sel = slot_team == t
            xt = x[:, sel]
            depth = np.where(sign[:, t:t + 1] > 0, xt, length - xt) / length       # 0 own goal, 1 opponent goal
            cols["cx_" + tag] = np.nanmean(depth, axis=1)
            cols["cy_" + tag] = np.nanmean(y[:, sel], axis=1) / width
            cols["sx_" + tag] = np.nanstd(depth, axis=1)
            cols["sy_" + tag] = np.nanstd(y[:, sel] / width, axis=1)
            srt = np.sort(np.where(np.isnan(depth), np.inf, depth), axis=1)
            cols["line2_" + tag] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)  # last outfield defender
            top = np.sort(np.where(np.isnan(depth), -np.inf, depth), axis=1)[:, -1]
            cols["front_" + tag] = np.where(np.isfinite(top), top, np.nan)
            cols["opp_half_" + tag] = (depth > 0.5).sum(axis=1).astype(np.float32)
            v = np.full(n, np.nan, dtype=np.float32)
            v[cfg.fps:] = cols["cx_" + tag][cfg.fps:] - cols["cx_" + tag][:-cfg.fps]   # push up over 1 s
            cols["push_" + tag] = v
    cols["cx_diff"] = cols["cx_h"] - cols["cx_a"]
    cols["sx_diff"] = cols["sx_h"] - cols["sx_a"]
    cols["sy_diff"] = cols["sy_h"] - cols["sy_a"]
    cols["push_diff"] = cols["push_h"] - cols["push_a"]
    for name in ("cx_diff", "sy_diff", "push_diff"):
        cols[name + "_3s"] = rolling_nanmean(cols[name], 3 * cfg.fps // 2)
    names = list(cols)
    return np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names]), names


def _fit(X, y, cfg: Config):
    """A deterministic LightGBM binary classifier (native API: no scikit-learn needed)."""
    import lightgbm as lgb
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "lambda_l2": 1.0,
              "seed": cfg.seed, "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=300)


def team_logit(players_xy, slot_team, attack, blocks, ctrl_team, cfg: Config):
    """(logit (N,) float32 log P(H)/P(A), report dict). Zero logits when LightGBM is missing or there
    are too few labelled frames to fit."""
    n = len(players_xy)
    out = np.zeros(n, dtype=np.float32)
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        log.warning("lightgbm not installed: no player-shape team evidence")
        return out, {"used": False}
    X, names = shape_features(players_xy, slot_team, attack, blocks, cfg)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b] = True
    labelled = graded & np.isin(ctrl_team, (0, 1))
    if labelled.sum() < 2000:
        return out, {"used": False, "labelled": int(labelled.sum())}
    yv = (ctrl_team == 1).astype(np.int8)                      # 1 = away
    held = labelled & ((np.arange(n) // BLOCK) % HOLD_EVERY == HOLD_EVERY - 1)
    train = labelled & ~held
    m = _fit(X[train], yv[train], cfg)
    acc = float(((m.predict(X[held]) > 0.5) == yv[held]).mean()) if held.any() else float("nan")
    m = _fit(X[labelled], yv[labelled], cfg)
    p = np.clip(m.predict(X[graded]), 1e-3, 1 - 1e-3)
    out[graded] = np.log((1 - p) / p).astype(np.float32)          # >0 favours home
    out = np.nan_to_num(rolling_nanmean(out, cfg.fps // 2)).astype(np.float32)   # +/- 0.5 s: no flicker
    report = {"used": True, "labelled": int(labelled.sum()), "held_out_accuracy": round(acc, 4),
              "share_called_home": round(float((p < 0.5).mean()), 3)}
    log.info("player-shape team model: %d labelled frames, held-out accuracy %.3f", report["labelled"], acc)
    return out, report
