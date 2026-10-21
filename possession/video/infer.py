"""Region-of-interest ball search: look only where the ball layer expects the ball."""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from . import VideoConfig
from .frames import crop_stack, gray_frames
from .geometry import pitch_to_px, px_per_metre, px_to_pitch
from .model import decode


def crop_sizes(prior_xy, prior_sigma, hs, cfg: Config, vcfg: VideoConfig):
    """Square search window per frame: the prior's uncertainty in pixels plus a margin, rounded up
    to a multiple of 32 (so frames can be batched by size), clipped to the allowed range."""
    radius = prior_sigma * px_per_metre(hs, prior_xy, cfg) + 24.0
    size = np.ceil(2.0 * radius / 32.0) * 32.0
    return np.clip(np.nan_to_num(size, nan=vcfg.infer_max_crop), vcfg.infer_min_crop,
                   vcfg.infer_max_crop).astype(int)


@torch.inference_mode()
def detect(video_path, frames, prior_xy, prior_sigma, homography, model, cfg: Config, vcfg: VideoConfig,
           device="cpu", expected_frames=None, log=None):
    """Search the video around each prior. Returns (xy metres (n, 2), confidence (n,)).

    Frames whose prior is too vague (sigma > infer_max_sigma_m) or undefined are not searched and
    keep confidence 0. The default device is the CPU on purpose: the submission has to reproduce
    exactly, and CPU inference of this small model is deterministic and fast enough.
    """
    model = model.to(device).eval()
    frames = np.asarray(frames, dtype=np.int64)
    n = len(frames)
    hs = homography[frames]
    centre = pitch_to_px(hs, prior_xy, cfg)
    size = crop_sizes(prior_xy, prior_sigma, hs, cfg, vcfg)
    lo, hi = min(vcfg.offsets), max(vcfg.offsets)
    ok = (prior_sigma <= vcfg.infer_max_sigma_m) & np.isfinite(centre).all(axis=1) & (frames + lo >= 0)

    order = [i for i in np.argsort(frames, kind="stable") if ok[i]]
    needed = sorted(set((frames[order][:, None] + np.array(vcfg.offsets)[None, :]).ravel().tolist()))
    conf = np.zeros(n)
    px = np.full((n, 2), np.nan)
    buckets = {}

    def flush(s):
        items = buckets.pop(s, [])
        if not items:
            return
        out = model(torch.from_numpy(np.stack([c for _, c, _, _ in items])).to(device))
        c, u, v = decode(out)
        for (i, _, x0, y0), cc, uu, vv in zip(items, c, u, v):
            conf[i], px[i] = cc, (x0 + uu, y0 + vv)

    buffers, ptr = {}, 0
    for idx, gray in gray_frames(video_path, needed, expected_frames):
        buffers[idx] = gray
        while ptr < len(order) and frames[order[ptr]] + hi <= idx:
            i = order[ptr]
            f = frames[i]
            crop, x0, y0 = crop_stack([buffers[f + o] for o in vcfg.offsets], centre[i, 0], centre[i, 1], size[i])
            buckets.setdefault(int(size[i]), []).append((i, crop, x0, y0))
            if len(buckets[int(size[i])]) >= vcfg.infer_batch:
                flush(int(size[i]))
            ptr += 1
            if log and ptr % 5000 == 0:
                log("  searched {} / {} frames".format(ptr, len(order)))
        for k in [k for k in buffers if ptr < len(order) and k < frames[order[ptr]] + lo]:
            del buffers[k]
    for s in list(buckets):
        flush(s)

    xy = np.full((n, 2), np.nan)
    found = conf > 0
    if found.any():
        xy[found] = px_to_pitch(hs[found], px[found], cfg)
    return xy, conf
