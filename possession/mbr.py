"""L6 -- minimum-Bayes-risk selection on the ranking metric (PLAN.md 5.7).

spellF1 punishes a spurious short possession and a missed one very differently from a per-frame
loss, so the single most likely segmentation is not necessarily the best answer. Draw posterior
segmentations, build candidate outputs (the MAP one, plus versions that drop possessions the
posterior is unsure about) and keep the candidate with the highest expected spellF1 against the
draws. `spell_f1` re-implements the official metric on spell lists; tests/test_mbr.py checks it
equals possession_metrics.spell_scores, which stays the only authority on the number.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
D = 2
Segment = Tuple[int, int, int]


def spells_of(blocks: Sequence[Sequence[Segment]]):
    """(start, stop, team) arrays of the possessions: every H or A segment, in time order."""
    rows = [(a, b, k) for segs in blocks for a, b, k in segs if k != D]
    rows.sort()
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    arr = np.array(rows, dtype=np.int64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def spell_f1(gt, pred, ious=IOUS):
    """Mean over IoU thresholds of the one-to-one greedy possession-matching F1 (official rules)."""
    gs, ge, gk = gt
    ps, pe, pk = pred
    n_gt, n_pred = len(gs), len(ps)
    if not n_gt or not n_pred:
        return 0.0
    inter = np.minimum(ge[:, None], pe[None, :]) - np.maximum(gs[:, None], ps[None, :])
    union = np.maximum(ge[:, None], pe[None, :]) - np.minimum(gs[:, None], ps[None, :])
    ok = (inter > 0) & (gk[:, None] == pk[None, :])
    gi, pj = np.nonzero(ok)
    if not len(gi):
        return 0.0
    iou = inter[gi, pj] / union[gi, pj].astype(np.float64)
    order = np.lexsort((pj, gi, -iou))
    iou, gi, pj = iou[order], gi[order], pj[order]
    scores = []
    for th in ious:
        used_g, used_p, tp = set(), set(), 0
        for v, g, p in zip(iou, gi, pj):
            if v < th:
                break
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            tp += 1
        scores.append(2.0 * tp / (n_gt + n_pred) if tp else 0.0)
    return float(np.mean(scores))


def _merge_adjacent(segs: List[list]):
    out = []
    for s in segs:
        if out and out[-1][2] == s[2]:
            out[-1][1] = s[1]
        else:
            out.append(list(s))
    return [tuple(s) for s in out]


def drop_spells(segments: Sequence[Segment], drop: Sequence[int]):
    """Absorb the listed segments into their predecessor (their successor for the first one)."""
    segs = [list(s) for s in segments]
    for i in sorted(drop):
        if i > 0:
            segs[i][2] = segs[i - 1][2]
        elif len(segs) > 1:
            segs[i][2] = segs[i + 1][2]
    return _merge_adjacent(segs)


def marginals(best: Sequence[Segment], samples: Sequence[Sequence[Segment]], iou_min=0.5):
    """For each alive segment of `best`: share of samples holding a same-team spell at IoU >= iou_min."""
    out = np.ones(len(best))
    if not samples:
        return out
    packed = []
    for s in samples:
        arr = np.array([x for x in s if x[2] != D], dtype=np.int64).reshape(-1, 3)
        packed.append(arr)
    for i, (a, b, k) in enumerate(best):
        if k == D:
            continue
        hits = 0
        for arr in packed:
            if not len(arr):
                continue
            inter = np.minimum(arr[:, 1], b) - np.maximum(arr[:, 0], a)
            union = np.maximum(arr[:, 1], b) - np.minimum(arr[:, 0], a)
            hits += bool(np.any((arr[:, 2] == k) & (inter > 0) & (inter >= iou_min * union)))
        out[i] = hits / float(len(packed))
    return out


def choose(results, cfg: Config):
    """results = [(half, best, samples)] from decode_match. Returns (segments per block, report)."""
    best_blocks = [r[1] for r in results]
    n_samples = len(results[0][2]) if results else 0
    if not n_samples:
        return best_blocks, {"theta": 0.0, "expected": None}

    draws = [spells_of([r[2][s] for r in results]) for s in range(n_samples)]
    marg = [marginals(r[1], r[2]) for r in results]
    report = {}
    best_theta, best_score, best_variant = None, -1.0, best_blocks
    for theta in cfg.mbr_thetas:
        variant = [drop_spells(r[1], [i for i, m in enumerate(mg) if r[1][i][2] != D and m < theta])
                   for r, mg in zip(results, marg)] if theta > 0 else best_blocks
        pred = spells_of(variant)
        score = float(np.mean([spell_f1(d, pred) for d in draws]))
        report[theta] = score
        if score > best_score + 1e-12:
            best_theta, best_score, best_variant = theta, score, variant
    return best_variant, {"theta": best_theta, "expected": best_score, "by_theta": report}
