"""Make clean tracking look like the automatic one (PLAN.md 5.5 "ATD simulator").

This module corrupts clean tracking the way the real ATD is corrupted: position jitter, bursty
loss of the ball, player drop-outs and spurious ball detections. The ball-loss model can be fitted
to a real gap histogram (`fit_gap_model`); the defaults only approximate a ball visible ~58% of
the time and have not been matched to the match data, which the acceptance test of PLAN.md 5.5
requires before external data can be trusted.
"""

from __future__ import annotations

import numpy as np

from .config import Config, DEFAULT
from .ingest import MatchInputs, Tracking
from .util import runs


def fit_gap_model(ball_xy):
    """Empirical visible-run and gap lengths of a real ball track (aggregate statistics only)."""
    seen = ~np.isnan(ball_xy[:, 0])
    s, e = runs(seen)
    g0, g1 = runs(~seen)
    return {"visible": (e - s).astype(int), "gap": (g1 - g0).astype(int)}


def ball_mask(n, rng, model=None, mean_visible=70.0, mean_gap=50.0):
    """Boolean visibility mask from alternating visible / missing runs."""
    mask = np.zeros(n, dtype=bool)
    t, visible = 0, bool(rng.randint(2))
    while t < n:
        if model is not None:
            pool = model["visible"] if visible else model["gap"]
            length = int(pool[rng.randint(len(pool))])
        else:
            length = int(rng.geometric(1.0 / (mean_visible if visible else mean_gap)))
        mask[t:t + length] = visible
        t += length
        visible = not visible
    return mask


def corrupt(match, cfg: Config = DEFAULT, seed=0, model=None, ball_noise=0.25, player_noise=0.2,
            outlier_rate=0.002, dropout_rate=1 / 3000.0, mean_dropout=40):
    """Return a Tracking built from a SyntheticMatch (or any clean arrays with the same fields)."""
    rng = np.random.RandomState(seed + 1000)
    n = len(match.ball_xy)
    players = match.players_xy.copy() + rng.randn(*match.players_xy.shape).astype(np.float32) * player_noise
    for slot in range(players.shape[1]):
        t = 0
        while t < n:
            t += int(rng.geometric(dropout_rate))
            players[t:t + int(rng.geometric(1.0 / mean_dropout)), slot] = np.nan

    ball = match.ball_xy.copy() + rng.randn(n, 2).astype(np.float32) * ball_noise
    visible = ball_mask(n, rng, model)
    spurious = visible & (rng.rand(n) < outlier_rate)
    ball[spurious] = rng.rand(int(spurious.sum()), 2) * np.array([cfg.pitch_length, cfg.pitch_width])
    ball[~visible] = np.nan
    speed = match.players_speed + rng.randn(*match.players_speed.shape).astype(np.float32) * 0.3
    speed[np.isnan(players[:, :, 0])] = np.nan
    return Tracking(players, np.maximum(speed, 0), ball, np.zeros((n, 0), dtype=np.float32))


def synthetic_tags(match, cfg: Config = DEFAULT, seed=0, wrong_rate=0.15):
    """Smart-Tagging-like intervals from the truth: noisy possession phases + set-piece windows."""
    rng = np.random.RandomState(seed + 2000)
    names = {0: "Barcelona", 1: "Atletic Club"}
    tags = []
    truth = match.truth
    edges = np.flatnonzero(np.diff(truth)) + 1
    bounds = np.concatenate(([0], edges, [len(truth)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        label = int(truth[a])
        if label == 2:
            tags.append({"code": "SET PIECES", "start_frame": max(0, a + int(1.0 * cfg.fps)),
                         "end_frame": min(len(truth), b + int(1.5 * cfg.fps)),
                         "tags": {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}})
        elif b - a > 5 * cfg.fps:
            who = label if rng.rand() > wrong_rate else 1 - label
            tags.append({"code": "BUILD UP", "start_frame": a + cfg.fps, "end_frame": b,
                         "tags": {"Team": names[who], "Half": "1st Half"}})
    return tags


def to_inputs(match, cfg: Config = DEFAULT, seed=0, with_tags=True, model=None) -> MatchInputs:
    """A full MatchInputs for the pipeline, over one graded block covering the whole match."""
    n = len(match.ball_xy)
    return MatchInputs(corrupt(match, cfg, seed, model), match.slot_team, [(1, 0, n)],
                       synthetic_tags(match, cfg, seed) if with_tags else [], kickoffs={1: 0},
                       n_frames=n)
