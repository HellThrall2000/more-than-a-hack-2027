"""Optional video ball detector (PLAN.md section 11, "ROI ball recovery").

The tracking-only pipeline loses the ball on ~41% of graded frames. This package recovers some of
those frames from the tactical video, without hand labelling and without leaving the machine:

  * training labels come for free -- on frames where the ATD sees the ball, the homography maps it
    to a pixel, so every visible ball is a training example (`dataset.py`);
  * a small fully-convolutional net (`model.py`) finds the ball in a crop, using three frames
    (t-3, t, t+3) so that motion separates it from static white pitch markings;
  * at inference the crop is centred on where the ball layer *expects* the ball and sized by its
    uncertainty, so only a small part of each frame is ever searched (`infer.py`);
  * `evaluate.py` hides visible frames and checks how often the detector recovers them, which is
    the go / no-go gate of PLAN.md section 11.

Needs OpenCV and PyTorch (requirements-video.txt); the rest of the package does not.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class VideoConfig:
    # training crops
    crop: int = 96                      # pixels, square
    offsets: tuple = (-3, 0, 3)         # frames stacked as channels
    stride: int = 8                     # one training frame per `stride` visible-ball frames
    pos_per_frame: int = 2              # crops with the ball at different places in the window
    neg_per_frame: int = 1              # crops around players, the ball well outside
    jitter_margin: int = 10             # the ball stays this far from the crop edge
    val_segment_frames: int = 3000      # contiguous blocks, so neighbouring frames never leak
    val_every: int = 5                  # every 5th block is held out
    # training
    epochs: int = 12
    batch: int = 128
    lr: float = 2e-3
    heat_sigma: float = 0.8             # gaussian target width, in output cells
    pixel_tol: float = 4.0              # a detection within this many pixels counts as correct
    gate_precision: float = 0.9         # PLAN.md 11: precision on masked visible frames
    # inference
    infer_min_crop: int = 96
    infer_max_crop: int = 320
    infer_max_sigma_m: float = 6.0      # beyond this the prior is too vague to search around
    infer_batch: int = 64
    seed: int = 0


STRIDE = 4                              # output cell size of the net, in input pixels
