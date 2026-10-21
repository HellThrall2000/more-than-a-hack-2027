"""Sequential video decoding and crop extraction (OpenCV).

Frames are read in order with grab()/retrieve(): seeking in H.264 is slow and inexact, and a frame
number that is off by one shifts every label. Only the frames that are actually needed are
converted to grey.
"""

from __future__ import annotations

import warnings

import numpy as np


def gray_frames(path, needed, expected_total=None):
    """Yield (index, grey image) for each index in the sorted sequence `needed`."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("cannot open video: {}".format(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if expected_total and total and total != expected_total:
        # FAQ.md: a different count means this is not the reference decode. Do not pad or trim.
        warnings.warn("video reports {} frames, expected {}; frame numbers may not line up with "
                      "the tracking".format(total, expected_total))
    idx, j, needed = -1, 0, list(needed)
    try:
        while j < len(needed):
            if not cap.grab():
                break
            idx += 1
            if idx == needed[j]:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, (frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                j += 1
                while j < len(needed) and needed[j] == idx:
                    j += 1
    finally:
        cap.release()


def crop_stack(frames, cx, cy, size):
    """(len(frames), size, size) uint8 window centred on (cx, cy), zero outside the image.

    Returns (crop, x0, y0) where (x0, y0) is the image position of the crop's top-left pixel, so
    a point at image (px, py) sits at (px - x0, py - y0) in the crop.
    """
    h, w = frames[0].shape
    x0, y0 = int(round(cx - size / 2.0)), int(round(cy - size / 2.0))
    out = np.zeros((len(frames), size, size), dtype=np.uint8)
    xs, xe, ys, ye = max(x0, 0), min(x0 + size, w), max(y0, 0), min(y0 + size, h)
    if xe > xs and ye > ys:
        for i, frame in enumerate(frames):
            out[i, ys - y0:ye - y0, xs - x0:xe - x0] = frame[ys:ye, xs:xe]
    return out, x0, y0
