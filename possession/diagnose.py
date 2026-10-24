"""Where does the pipeline lose points? Aggregate counts only -- safe to paste into a chat.

    python -m possession.diagnose [--gt public|PATH]

Runs the pipeline on the real bundle and compares it with a ground truth (the public 5 minutes by
default, or a hand-labelled window) layer by layer:

  * the confusion matrix and the alive / dead and team scores,
  * emission alone (per-frame argmax, no decoder) against the decoded labels,
  * how often a candidate boundary exists near each true transition -- if it does not, no decoder
    setting can place the boundary,
  * what the ball layer saw inside the window, and how the dead-ball flags line up.

Use it to find BUGS (swapped teams, a dead ball never detected, a lag). It must not be used to tune
thresholds on the public window: 17 possessions cannot rank two settings (EVALUATION.md).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import log as plog
from .ball import SRC_NAMES
from .config import DEFAULT, Config
from .policy import metrics

log = plog.get(__name__)
NAMES = ("H", "A", "D")


def _near(frames, targets):
    """Distance from each target frame to the closest element of `frames` (inf if none)."""
    if not len(frames):
        return np.full(len(targets), np.inf)
    frames = np.sort(frames)
    k = np.clip(np.searchsorted(frames, targets), 1, len(frames) - 1)
    return np.minimum(np.abs(frames[k] - targets), np.abs(frames[k - 1] - targets)) if len(frames) > 1 \
        else np.abs(frames[0] - targets)


def report(labels, details, gt, cfg: Config = DEFAULT):
    """The diagnostic text for one prediction. `gt` is an int8 code array (0 H, 1 A, 2 D, 3 X)."""
    pm = metrics()
    pred = pm.label_codes(labels)
    res = pm.evaluate(gt, pred)
    win = gt != pm.X
    out = []
    p = out.append

    p("window: {} graded frames, {} gt possessions, {} predicted".format(
        int(win.sum()), res["n_gt_spells"], res["n_pred_spells"]))
    p("spellF1 {:.4f}   macroF1 {:.4f}   alive/dead F1 {:.4f}   team acc (both alive) {:.4f}".format(
        res["spell_f1_mean"], res["macro_f1"], res["alive_dead_f1"], res["team_acc_both_alive"]))
    p("spellF1 by IoU: " + "  ".join("@{:.1f} {:.3f}".format(i, res["spell_f1"][i]["f1"]) for i in pm.SPELL_IOUS))
    p("\nconfusion (rows = gt, cols = predicted)        H       A       D       X")
    for k, name in enumerate(NAMES):
        p("  gt {}                                 {}".format(name, "".join("{:8d}".format(v) for v in res["confusion"][k])))
    for name in NAMES:
        c = res["per_class"][name]
        p("  {}: precision {:.3f}  recall {:.3f}  F1 {:.3f}  support {}".format(
            name, c["precision"], c["recall"], c["f1"], c["support"]))

    # emission alone: is the per-frame evidence already wrong, or does the decoder lose it?
    logp = details["logp"]
    em = np.argmax(logp, axis=1)
    em_codes = np.where(win, em, pm.X).astype(np.int8)
    em_res = pm.evaluate(gt, em_codes)
    p("\nemission argmax with NO decoder: macroF1 {:.4f}  team acc (both alive) {:.4f}  alive/dead F1 {:.4f}".format(
        em_res["macro_f1"], em_res["team_acc_both_alive"], em_res["alive_dead_f1"]))

    # team bias: where does it start? Each source's team call on the frames that are truly H or A.
    p("\nteam calls on truly-alive frames, split by the true team (share called H / called A)")
    pred_codes = pm.label_codes(labels)
    sources = [("decoded output", pred_codes),
               ("emission argmax", np.argmax(logp, axis=1))]
    tagf = details.get("tagf")
    if tagf is not None:
        tp = tagf["poss"]
        tag_call = np.where(tp[:, 0] > tp[:, 1], 0, np.where(tp[:, 1] > tp[:, 0], 1, -1))
        sources.append(("Smart Tagging alone", tag_call))
    ctrl = details["ball"].ctrl_team
    sources.append(("ball control alone", ctrl))
    for name, call in sources:
        cells = []
        for team, tname in ((0, "H"), (1, "A")):
            sel = win & (gt == team) & (call >= 0)
            n_team = int((win & (gt == team)).sum())
            if not sel.any():
                cells.append("true {}: no calls".format(tname))
                continue
            cells.append("true {} (covers {:.0%}): H {:.0%} A {:.0%}{}".format(
                tname, sel.sum() / max(n_team, 1), float((call[sel] == 0).mean()), float((call[sel] == 1).mean()),
                "  D {:.0%}".format(float((call[sel] == 2).mean())) if (call[sel] == 2).any() else ""))
        p("  {:<22} {}".format(name, " | ".join(cells)))
    p("  a source that leans one way on BOTH rows is biased; compare runs with --set emit_tag_weight=0")

    # candidate recall: does a boundary exist near every true transition?
    cands = details["cands"]
    runs = pm.valid_runs(gt)
    gt_tr = pm.transitions(gt, runs)
    kinds = {"team change (H<->A)": [], "alive -> D": [], "D -> alive": []}
    for f, a, b in gt_tr:
        key = "alive -> D" if b == 2 else ("D -> alive" if a == 2 else "team change (H<->A)")
        kinds[key].append(f)
    p("\ncandidate boundaries near true transitions (any candidate / one carrying evidence)")
    for key, frames in kinds.items():
        if not frames:
            continue
        frames = np.array(frames)
        bonus = {"team change (H<->A)": cands.turn, "alive -> D": cands.stop, "D -> alive": cands.restart}[key]
        d_any = _near(cands.frame, frames)
        d_ev = _near(cands.frame[bonus > 0], frames)
        p("  {:<20} n={:<3} within 12 fr: {:>2}/{:<2}  within 25 fr: {:>2}/{:<2}   | with evidence: 12 fr {:>2}  25 fr {:>2}".format(
            key, len(frames), int((d_any <= 12).sum()), len(frames), int((d_any <= 25).sum()), len(frames),
            int((d_ev <= 12).sum()), int((d_ev <= 25).sum())))

    t = res["transitions"][25]
    lat = t["mean_latency"]
    p("\ntransitions at +/-25 frames: gt {}  matched {}  missed {}  spurious {}  mean latency {}".format(
        t["n_gt"], t["n_matched"], t["n_missed"], t["n_spurious"], "--" if lat is None else "{:+.1f} fr".format(lat)))
    for label, group in (("missed", t["missed"]), ("spurious", t["spurious"])):
        tally = {}
        for _, a, b in group:
            key = "{}->{}".format(pm.LABELS[a], pm.LABELS[b])
            tally[key] = tally.get(key, 0) + 1
        p("  {:<9} by type: {}".format(label, tally if tally else "none"))

    # what the ball layer saw in the window
    ball = details["ball"]
    p("\nball layer inside the window")
    p("  frame share by source: " + "  ".join("{} {:.0%}".format(SRC_NAMES[s], float((ball.src[win] == s).mean()))
                                              for s in range(len(SRC_NAMES)) if (ball.src[win] == s).any()))
    p("  control episodes cover {:.0%} of window frames; {} touches detected".format(
        float((ball.ctrl_team[win] >= 0).mean()), int(((ball.touches["touch_frame"] >= np.flatnonzero(win)[0])
                                                       & (ball.touches["touch_frame"] <= np.flatnonzero(win)[-1])).sum())))
    dead, alive = win & (gt == 2), win & (gt < 2)
    for name, flag in (("oob", ball.oob), ("stationary", ball.stationary)):
        p("  {:<10} flagged on {:.0%} of gt-dead frames and {:.0%} of gt-alive frames".format(
            name, float(flag[dead].mean()) if dead.any() else 0.0, float(flag[alive].mean()) if alive.any() else 0.0))
    ctrl = ball.ctrl_team
    sel = alive & (ctrl >= 0)
    if sel.any():
        p("  control team == gt team on {:.0%} of {} alive frames with a control episode".format(
            float((ctrl[sel] == gt[sel]).mean()), int(sel.sum())))
    p("  Smart Tagging weights learned: {}".format({k: round(v, 2) for k, v in details["tag_weights"].items()}))
    p("  MBR picked theta {}".format(details["mbr"]["theta"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", default="public")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a Config value, e.g. --set emit_tag_weight=0 (repeatable)")
    args = ap.parse_args(argv)
    plog.setup("diagnose")

    from .harness import resolve_gt
    from .ingest import load_match
    from .run import predict

    cfg = DEFAULT.with_overrides(args.set)
    if args.set:
        log.info("config overrides %s -> config %s", args.set, cfg.digest())
    pm = metrics()
    gt = pm.label_codes(pm.read_label_csv(resolve_gt(args.gt)))
    with plog.stage("load the match"):
        inputs = load_match(cfg, args.data_dir)
    with plog.stage("predict"):
        labels, details = predict(inputs, cfg, details=True)
    log.info("diagnosis against %s:\n%s", args.gt, report(labels, details, gt, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
