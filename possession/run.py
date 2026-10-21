"""End-to-end pipeline: ingest -> players -> ball state -> features -> emission -> decode -> MBR.

    python -m possession.run --out submissions/<team>.csv                 # real match, ./data
    python -m possession.run --synthetic --seed 0                          # no data needed

`predict` takes a MatchInputs, so the same code path serves the real bundle, the synthetic fixture
and any future external match. It prints only aggregates, never frame-level data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import ROOT
from .ball import build_ball_state
from .candidates import build_candidates
from .config import Config, DEFAULT
from .decode import decode_match
from .emission_rules import emit
from .features import build_features
from .ingest import MatchInputs
from .mbr import choose
from .players import attack_direction, clean_players, descriptors
from .policy import render, validate, write_csv
from . import tags as tagmod


def predict(inputs: MatchInputs, cfg: Config = DEFAULT, verbose=False, details=False):
    """Labels (N,) of 'H' / 'A' / 'D' / 'X' for one match. With details=True also the internals."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()
    tr, slot_team, blocks = inputs.tracking, inputs.slot_team, inputs.blocks
    n = inputs.n_frames

    players = clean_players(tr.players_xy, slot_team, cfg)
    desc = descriptors(players, tr.players_speed, slot_team, cfg)
    attack = attack_direction(players, slot_team, blocks)
    ball = build_ball_state(tr.ball_xy, players, slot_team, blocks, cfg, inputs.video_ball)
    log("L1-L2 ball state      %.1fs" % (time.time() - t0))

    fs, aux = build_features(ball, players, desc, slot_team, attack, blocks, cfg)
    log("L3 features           %.1fs  (%d columns)" % (time.time() - t0, len(fs.names)))

    reliable = np.isin(ball.src, (0, 1))
    silver = tagmod.silver_team(aux["dist"], reliable, cfg)
    weights, counts = tagmod.calibrate(inputs.tags, silver, cfg)
    tagf = tagmod.build(inputs.tags, n, weights, cfg)
    logp = emit(fs, tagf, cfg)
    cands = build_candidates(ball, fs, aux, blocks, inputs.kickoffs, cfg)
    log("L3-L4 emission/cands  %.1fs  (%d candidates)" % (time.time() - t0, len(cands)))

    results = decode_match(logp, cands, blocks, inputs.kickoffs, cfg, n_samples=cfg.mbr_samples)
    segments, report = choose(results, cfg)
    log("L5-L6 decode + MBR    %.1fs  (theta=%s)" % (time.time() - t0, report["theta"]))

    labels = render(n, segments)
    if details:
        return labels, {"ball": ball, "features": fs, "logp": logp, "cands": cands,
                        "results": results, "segments": segments, "mbr": report,
                        "tag_weights": weights, "tag_counts": counts, "players": players}
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    ap.add_argument("--data-dir", default=None, help="defaults to ./data or MTAH_DATA_DIR")
    ap.add_argument("--synthetic", action="store_true", help="run on the synthetic fixture instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--video-ball", default=None, help="detections from possession.video.detect")
    args = ap.parse_args(argv)
    cfg = DEFAULT

    if args.synthetic:
        from .simulate_atd import to_inputs
        from .synthetic import make_match
        match = make_match(9000, seed=args.seed, cfg=cfg)
        inputs = to_inputs(match, cfg, seed=args.seed)
        labels = predict(inputs, cfg, verbose=True)
        pm = __import__("possession.policy", fromlist=["metrics"]).metrics()
        truth = np.array(["H", "A", "D"])[match.truth]
        r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))
        print("synthetic: macroF1 %.4f  spellF1 %.4f  possessions %d vs %d" % (
            r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
        return 0

    from .ingest import load_match
    inputs = load_match(cfg, args.data_dir, use_cache=not args.no_cache)
    if args.video_ball:
        z = np.load(args.video_ball)
        inputs.video_ball = (z["xy"], z["conf"], float(z["thr"]))
    labels = predict(inputs, cfg, verbose=True)
    reason = validate(labels, inputs.n_frames)
    if reason:
        print("INVALID:", reason)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_csv(args.out, labels)
    print("wrote", args.out, " -- now run: python evaluation/validate_submission.py", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
