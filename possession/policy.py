"""L6 -- render segments to the submission and check it (PLAN.md 5.7)."""

from __future__ import annotations

import os
import sys

import numpy as np

from . import EVAL_DIR, ROOT

CHAR = {0: "H", 1: "A", 2: "D"}


def render(n_frames, blocks_segments):
    """(N,) array of 'H' / 'A' / 'D', 'X' on every frame outside the decoded blocks.

    Graded frames never get an X: spellF1 never sees one (it would be forward-filled) and macroF1
    counts it as a miss, so abstaining on a graded frame cannot help the ranking metric.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    for segs in blocks_segments:
        for a, b, k in segs:
            labels[a:b] = CHAR[int(k)]
    return labels


def write_csv(path, labels):
    from starter.baseline import write_submission
    return write_submission(path, labels)


def metrics():
    """The official metric module, imported from evaluation/ (never re-implemented here)."""
    if EVAL_DIR not in sys.path:
        sys.path.insert(0, EVAL_DIR)
    import possession_metrics as pm
    return pm


def validate(labels, n_expected=None):
    """None if `labels` would pass validate_submission, else the one-line reason."""
    pm = metrics()
    import pandas as pd
    n = n_expected or len(labels)
    return pm.validate_submission(pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}), n)
