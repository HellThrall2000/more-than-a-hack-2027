"""Validation harness (PLAN.md 7): score any label file with the OFFICIAL metric, log the result.

    python -m possession.harness submissions/x.csv                      # public 5 minutes
    python -m possession.harness submissions/x.csv --gt labels/win1.csv  # a hand-labelled window
    python -m possession.harness submissions/x.csv --bootstrap 1000

The ground truth is any `frame,label` CSV with X outside the graded span, so the public window and
hand-labelled validation windows (V3) go through the same path. The public window checks that the
pipeline is correct; it must never choose between two versions (EVALUATION.md). Every call is
appended to a registry keyed by config hash, which is where the ablation table comes from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import ROOT
from . import log as plog
from .config import DEFAULT
from .policy import metrics

log = plog.get(__name__)

REGISTRY = os.path.join(ROOT, "cache", "registry.jsonl")
KEYS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "mean_spell_duration_s",
        "gt_mean_spell_duration_s", "spell_bootstrap_low", "spell_bootstrap_high", "alive_dead_f1",
        "team_acc_both_alive", "n_graded")


def resolve_gt(gt):
    """'public' | 'private' | path -> a path, using the organisers' own resolver."""
    import run_evaluation
    return run_evaluation.find_gt(gt)


def score(labels, gt="public", bootstrap=0):
    """Official metrics for an array of 'H'/'A'/'D'/'X' labels against a ground-truth CSV."""
    pm = metrics()
    gt_codes = pm.label_codes(pm.read_label_csv(resolve_gt(gt)))
    return pm.evaluate(gt_codes, pm.label_codes(labels), bootstrap_n=bootstrap)


def score_csv(path, gt="public", bootstrap=0):
    pm = metrics()
    table = pm.read_label_csv(path)
    gt_codes = pm.label_codes(pm.read_label_csv(resolve_gt(gt)))
    reason = pm.validate_submission(table, len(gt_codes))
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    return pm.evaluate(gt_codes, pm.label_codes(table), bootstrap_n=bootstrap)


def log_result(result, name, gt, cfg=DEFAULT, path=REGISTRY):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name, "gt": os.path.basename(str(gt)),
           "config": cfg.digest()}
    row.update({k: result.get(k) for k in KEYS})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("submission")
    ap.add_argument("--gt", default="public")
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args(argv)
    plog.setup("harness")
    result = score_csv(args.submission, args.gt, args.bootstrap)
    row = log_result(result, os.path.basename(args.submission), args.gt)
    log.info("{name}: spellF1 {s:.4f}  macroF1 {m:.4f}  possessions {p} (gt {g})  config {c}".format(
        name=row["name"], s=result["spell_f1_mean"], m=result["macro_f1"],
        p=result["n_pred_spells"], g=result["n_gt_spells"], c=row["config"]))
    if result["spell_bootstrap_low"] is not None:
        log.info("  spellF1 bootstrap 5-95%%: %.4f .. %.4f", result["spell_bootstrap_low"],
                 result["spell_bootstrap_high"])
    if str(args.gt) == "public":
        log.info("  (public window: 17 possessions -- a correctness check, never a way to pick a version)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
