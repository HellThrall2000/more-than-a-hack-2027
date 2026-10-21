"""Training crops with free labels: the ATD ball, projected through the homography (no hand labels)."""

from __future__ import annotations

import dataclasses

import numpy as np

from ..config import Config
from . import VideoConfig
from .frames import crop_stack, gray_frames
from .geometry import pitch_to_px


@dataclasses.dataclass
class CropSet:
    X: np.ndarray           # (n, C, S, S) uint8
    uv: np.ndarray          # (n, 2) float32, ball position inside the crop (pixels); NaN if absent
    present: np.ndarray     # (n,) bool
    frame: np.ndarray       # (n,) int32, video frame

    def save(self, path):
        np.savez_compressed(path, X=self.X, uv=self.uv, present=self.present, frame=self.frame)

    @staticmethod
    def load(path):
        z = np.load(path)
        return CropSet(z["X"], z["uv"], z["present"], z["frame"])


def sample_frames(ball_clean, blocks, vcfg: VideoConfig, rng):
    """Frames with a reliably seen ball, one in every `stride`, away from the block edges."""
    lo, hi = -min(vcfg.offsets), max(vcfg.offsets)
    out = []
    for _, a, b in blocks:
        for f in range(a + lo + rng.randint(vcfg.stride), b - hi, vcfg.stride):
            if not np.isnan(ball_clean[f, 0]):
                out.append(f)
    return np.array(out, dtype=np.int64)


def build(video_path, ball_clean, players_xy, homography, blocks, cfg: Config, vcfg: VideoConfig,
          expected_frames=None, log=None):
    """Positive crops (ball somewhere in the window) and negatives (centred on players, ball outside)."""
    rng = np.random.RandomState(vcfg.seed)
    sample = sample_frames(ball_clean, blocks, vcfg, rng)
    size, half = vcfg.crop, vcfg.crop / 2.0
    ball_px = pitch_to_px(homography[sample], ball_clean[sample], cfg)
    needed = sorted(set((sample[:, None] + np.array(vcfg.offsets)[None, :]).ravel().tolist()))

    cap = len(sample) * (vcfg.pos_per_frame + vcfg.neg_per_frame)
    X = np.empty((cap, len(vcfg.offsets), size, size), dtype=np.uint8)     # preallocated: a list + stack would double the peak
    count = 0
    uv, present, frame = [], [], []
    buffers, ptr = {}, 0
    lo, hi = min(vcfg.offsets), max(vcfg.offsets)
    for idx, gray in gray_frames(video_path, needed, expected_frames):
        buffers[idx] = gray
        while ptr < len(sample) and sample[ptr] + hi <= idx:
            f = sample[ptr]
            stack = [buffers[f + o] for o in vcfg.offsets]
            h, w = stack[0].shape
            bx, by = ball_px[ptr]
            if -vcfg.jitter_margin <= bx < w + vcfg.jitter_margin and -vcfg.jitter_margin <= by < h + vcfg.jitter_margin:
                span = half - vcfg.jitter_margin
                for _ in range(vcfg.pos_per_frame):
                    jx, jy = rng.uniform(-span, span, 2)
                    crop, x0, y0 = crop_stack(stack, bx + jx, by + jy, size)
                    X[count] = crop
                    count += 1
                    uv.append((bx - x0, by - y0))
                    present.append(True)
                    frame.append(f)
                valid = ~np.isnan(players_xy[f, :, 0])
                far = valid & (np.hypot(players_xy[f, :, 0] - ball_clean[f, 0],
                                        players_xy[f, :, 1] - ball_clean[f, 1]) > 8.0)
                cand = np.flatnonzero(far)
                for _ in range(vcfg.neg_per_frame if len(cand) else 0):
                    s = cand[rng.randint(len(cand))]
                    qx, qy = pitch_to_px(homography[f:f + 1], players_xy[f, s][None, :], cfg)[0]
                    jx, jy = rng.uniform(-24, 24, 2)
                    crop, x0, y0 = crop_stack(stack, qx + jx, qy + jy, size)
                    if x0 - 4 <= bx <= x0 + size + 4 and y0 - 4 <= by <= y0 + size + 4:
                        continue        # the ball is in or near this window: not a safe negative
                    X[count] = crop
                    count += 1
                    uv.append((np.nan, np.nan))
                    present.append(False)
                    frame.append(f)
            ptr += 1
            if log and ptr % 2000 == 0:
                log("  {} / {} frames".format(ptr, len(sample)))
        for k in [k for k in buffers if ptr < len(sample) and k < sample[ptr] + lo]:
            del buffers[k]
    if not count:
        raise RuntimeError("no training crops produced: wrong video, or no ball in the tracking?")
    return CropSet(X[:count], np.array(uv, dtype=np.float32), np.array(present),
                   np.array(frame, dtype=np.int32))
