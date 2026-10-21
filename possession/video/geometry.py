"""Pitch <-> pixel mappings with a different homography per frame (vectorised)."""

from __future__ import annotations

import numpy as np

from ..config import Config


def project_many(hs, x, y):
    """Apply matrices hs (n, 3, 3) to points (x, y) of shape (n,)."""
    den = hs[:, 2, 0] * x + hs[:, 2, 1] * y + hs[:, 2, 2]
    return ((hs[:, 0, 0] * x + hs[:, 0, 1] * y + hs[:, 0, 2]) / den,
            (hs[:, 1, 0] * x + hs[:, 1, 1] * y + hs[:, 1, 2]) / den)


def pitch_to_px(hs, xy_m, cfg: Config):
    """Metres on the pitch -> pixels, for matrices hs (n, 3, 3) and positions xy_m (n, 2)."""
    px, py = project_many(hs, xy_m[:, 0] / cfg.pitch_length, xy_m[:, 1] / cfg.pitch_width)
    return np.column_stack([px, py])


def px_to_pitch(hs, px_xy, cfg: Config):
    """Pixels -> metres on the pitch (ground plane: a ball in the air lands beyond where it is)."""
    nx, ny = project_many(np.linalg.inv(hs), px_xy[:, 0], px_xy[:, 1])
    return np.column_stack([nx * cfg.pitch_length, ny * cfg.pitch_width])


def px_per_metre(hs, xy_m, cfg: Config):
    """Local image scale at each position: pixels per metre along the two pitch axes, averaged."""
    p0 = pitch_to_px(hs, xy_m, cfg)
    p1 = pitch_to_px(hs, xy_m + np.array([1.0, 0.0]), cfg)
    p2 = pitch_to_px(hs, xy_m + np.array([0.0, 1.0]), cfg)
    return 0.5 * (np.hypot(*(p1 - p0).T) + np.hypot(*(p2 - p0).T))
