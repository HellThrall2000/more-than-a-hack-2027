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
from . import log as plog
from .ball import SRC_NAMES, build_ball_state
from .candidates import build_candidates
from .config import Config, DEFAULT
from .decode import decode_match
from .emission_rules import emit
from .features import build_features
from .ingest import MatchInputs
from .mbr import choose
from .players import attack_direction, clean_players, descriptors
from .restarts import kickoff_team_from_players, resolve_kickoff_team
from .policy import render, validate, write_csv
from . import tags as tagmod

log = plog.get(__name__)


def _teams(by_half):
    return {h: "HA"[t] for h, t in sorted((by_half or {}).items())} or "unknown"


def predict(inputs: MatchInputs, cfg: Config = DEFAULT, details=False):
    """Labels (N,) of 'H' / 'A' / 'D' / 'X' for one match. With details=True also the internals.

    Logs one timed line per layer, with the aggregate counts that show whether it did something
    sensible; nothing is printed unless the caller set up logging (possession.log.setup).
    """
    t0 = time.time()
    tr, slot_team, blocks = inputs.tracking, inputs.slot_team, inputs.blocks
    n = inputs.n_frames
    graded = np.zeros(n, dtype=bool)
    for _, a, b in blocks:
        graded[a:b] = True

    players = clean_players(tr.players_xy, slot_team, cfg)
    desc = descriptors(players, tr.players_speed, slot_team, cfg)
    attack = attack_direction(players, slot_team, blocks)
    log.info("L1 players            %6.1fs", time.time() - t0)
    ball = build_ball_state(tr.ball_xy, players, slot_team, blocks, cfg, inputs.video_ball)
    src = ball.src[graded]
    log.info("L2 ball state         %6.1fs  graded frames by source: %s", time.time() - t0,
             "  ".join("{} {:.1%}".format(name, float((src == k).mean()))
                       for k, name in enumerate(SRC_NAMES) if (src == k).any()))
    log.info("                               %d touches, control on %.1f%%, out of play on %.1f%%, ball at rest "
             "on %.1f%%%s", len(ball.touches["touch_frame"]), 100.0 * float((ball.ctrl_team[graded] >= 0).mean()),
             100.0 * float(ball.oob[graded].mean()), 100.0 * float(ball.stationary[graded].mean()),
             "" if inputs.video_ball is None else
             ", video detections used on {} frames".format(int(ball.video_used.sum())))

    fs, aux = build_features(ball, players, desc, slot_team, attack, blocks, cfg)
    log.info("L3 features           %6.1fs  %d columns", time.time() - t0, len(fs.names))

    reliable = np.isin(ball.src, (0, 1))
    reference = tagmod.silver_team(aux["dist"], reliable, cfg) if cfg.tag_calib == "silver" else ball.ctrl_team
    weights, counts = tagmod.calibrate(inputs.tags, reference, cfg)
    log.info("Smart Tagging weights (%s calibration): %s", cfg.tag_calib,
             {k: round(v, 2) for k, v in sorted(weights.items())})
    log.debug("Smart Tagging (agree, disagree) %s: %s", "frames" if cfg.tag_calib == "silver" else "intervals", counts)
    tagf = tagmod.build(inputs.tags, n, weights, cfg)
    logp = emit(fs, tagf, cfg)
    em = np.argmax(logp[graded], axis=1)
    cands = build_candidates(ball, fs, aux, blocks, inputs.kickoffs, cfg)
    log.info("L4 emission + cands   %6.1fs  per-frame argmax H %.1f%% A %.1f%% D %.1f%% | %d candidates "
             "(%d turnover, %d stoppage, %d restart)", time.time() - t0,
             *(100.0 * float((em == k).mean()) for k in range(3)), len(cands),
             int((cands.turn > 0).sum()), int((cands.stop > 0).sum()), int((cands.restart > 0).sum()))

    from_players = kickoff_team_from_players(players, slot_team, inputs.kickoffs, cfg)
    kickoff_team = resolve_kickoff_team(inputs.kickoff_team, from_players)
    log.info("kick-off team by half: tags %s, tracking %s -> %s", _teams(inputs.kickoff_team),
             _teams(from_players), _teams(kickoff_team))
    results = decode_match(logp, cands, blocks, inputs.kickoffs, cfg, n_samples=cfg.mbr_samples,
                           kickoff_team=kickoff_team)
    segments, report = choose(results, cfg)
    log.info("L5-L6 decode + MBR    %6.1fs  theta %s", time.time() - t0, report["theta"])

    labels = render(n, segments)
    lab = labels[graded]
    n_poss = sum(1 for segs in segments for a, b, k in segs if k != 2)
    n_dead = sum(1 for segs in segments for a, b, k in segs if k == 2)
    log.info("result: %d possessions (mean %.1f s), %d dead-ball stretches | graded frames H %.1f%% A %.1f%% "
             "D %.1f%%", n_poss, float(np.isin(lab, ("H", "A")).sum()) / max(n_poss, 1) / cfg.fps, n_dead,
             *(100.0 * float((lab == c).mean()) for c in "HAD"))
    if details:
        return labels, {"ball": ball, "features": fs, "aux": aux, "logp": logp, "cands": cands, "tagf": tagf,
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
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a Config value, e.g. --set emit_tag_weight=0 (repeatable)")
    args = ap.parse_args(argv)
    cfg = DEFAULT.with_overrides(args.set)
    plog.setup("run_synthetic" if args.synthetic else "run")
    if args.set:
        log.info("config overrides %s -> config %s", args.set, cfg.digest())

    if args.synthetic:
        from .simulate_atd import to_inputs
        from .synthetic import make_match
        match = make_match(9000, seed=args.seed, cfg=cfg)
        inputs = to_inputs(match, cfg, seed=args.seed)
        labels = predict(inputs, cfg)
        pm = __import__("possession.policy", fromlist=["metrics"]).metrics()
        truth = np.array(["H", "A", "D"])[match.truth]
        r = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))
        log.info("synthetic: macroF1 %.4f  spellF1 %.4f  possessions %d vs %d" % (
            r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))
        return 0

    from .ingest import load_match
    with plog.stage("load the match"):
        inputs = load_match(cfg, args.data_dir, use_cache=not args.no_cache)
    if args.video_ball:
        z = np.load(args.video_ball)
        inputs.video_ball = (z["xy"], z["conf"], float(z["thr"]))
        log.info("video detections from %s (threshold %.2f)", args.video_ball, float(z["thr"]))
    with plog.stage("predict (config {})".format(cfg.digest())):
        labels = predict(inputs, cfg)
    reason = validate(labels, inputs.n_frames)
    if reason:
        log.error("INVALID submission: %s", reason)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_csv(args.out, labels)
    log.info("wrote %s -- now run: python evaluation/validate_submission.py %s", args.out, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
