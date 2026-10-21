"""L2 ball-state reconstruction (PLAN.md 5.3): clean -> smooth -> gaps -> events -> flags."""

from __future__ import annotations

import dataclasses

import numpy as np

from ..config import Config
from . import clean as _clean
from . import events as _events
from . import gaps as _gaps
from . import smooth as _smooth
from .gaps import (SRC_CARRIED, SRC_FLIGHT, SRC_INTERP, SRC_NAMES, SRC_NONE, SRC_OBSERVED,  # noqa: F401
                   SRC_UNKNOWN)


@dataclasses.dataclass
class BallState:
    xy: np.ndarray
    sigma: np.ndarray
    src: np.ndarray
    vel: np.ndarray               # (N, 2) m/s everywhere (from the reconstructed track in gaps)
    speed: np.ndarray
    gaps: _gaps.Gaps
    ctrl_team: np.ndarray         # control from observed frames only
    carrier_team: np.ndarray      # weaker evidence from carried gaps
    touches: dict
    oob: np.ndarray
    stationary: np.ndarray
    observed: np.ndarray
    video_used: np.ndarray = None   # frames whose position came from the video detector


def merge_video(ball_xy, video, cfg: Config):
    """Add video detections to the ATD ball track on frames where the ATD has none.

    `video` is (xy, conf, thr): detections at or above the threshold validated on held-out frames.
    They enter as ordinary observations, so the speed gate in clean() still rejects any that are
    inconsistent with the surrounding track.
    """
    xy, conf, thr = video
    use = (np.isnan(ball_xy[:, 0]) & (conf >= max(thr, cfg.video_conf_min)) & np.isfinite(xy).all(axis=1))
    merged = ball_xy.copy()
    merged[use] = xy[use]
    return merged, use


def build_ball_state(ball_xy, players_xy, slot_team, blocks, cfg: Config, video=None) -> BallState:
    video_used = np.zeros(len(ball_xy), dtype=bool)
    if video is not None:
        ball_xy, video_used = merge_video(ball_xy, video, cfg)
    cleaned = _clean.clean(ball_xy, cfg)
    pos, vel, observed, interp = _smooth.smooth_ball(cleaned, cfg)
    gaps = _gaps.reconstruct(pos, vel, observed & ~np.isnan(pos[:, 0]), interp, players_xy,
                             slot_team, blocks, cfg)

    reliable = np.isin(gaps.src, (SRC_OBSERVED, SRC_INTERP))
    rel_pos = np.where(reliable[:, None], gaps.xy, np.nan)
    rel_vel = np.where(reliable[:, None], vel, np.nan)
    ev = _events.control_and_touches(rel_pos, rel_vel, players_xy, slot_team, cfg)

    # velocity of the reconstructed track, also inside gaps
    full_vel = np.full_like(gaps.xy, np.nan)
    full_vel[1:-1] = (gaps.xy[2:] - gaps.xy[:-2]) * (cfg.fps / 2.0)
    full_vel = np.where(reliable[:, None], vel, full_vel)
    speed = np.hypot(full_vel[:, 0], full_vel[:, 1])

    oob, stationary = _events.out_of_play(gaps.xy, gaps.src, gaps.sigma, full_vel, cfg)
    return BallState(gaps.xy, gaps.sigma, gaps.src, full_vel, speed, gaps, ev["ctrl_team"],
                     gaps.carrier_team, ev, oob, stationary, observed, video_used)
