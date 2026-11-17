"""The aggregate record of one scripts/run_all.sh round -> benchmarks/results/<UTC>_run-all.json.

    python -m possession.run_record --label TotalFootball --steps logs/run_all_<t>/steps.tsv
        --summary reports/summary_<t>.txt

Timings, software and hardware versions, git state, step status and the public-window scores: the
things benchmarks/README.md allows in that folder. It reads only the run's step table and its
aggregate-only summary, never a submission or a frame list.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

from . import ROOT
from . import log as plog

log = plog.bind(__name__)

SCORE = re.compile(r"(?P<file>[\w.]+?)\.csv: spellF1 (?P<spell_f1>[0-9.]+)\s+macroF1 (?P<macro_f1>[0-9.]+)\s+"
                   r"possessions (?P<possessions>\d+) \(gt (?P<gt_possessions>\d+)\)\s+config (?P<config>\w+)")
BOOT = re.compile(r"bootstrap 5-95%: (?P<lo>[0-9.]+) \.\. (?P<hi>[0-9.]+)")
RESULT = re.compile(r"result: (?P<possessions>\d+) possessions \(mean (?P<mean_s>[0-9.]+) s\), (?P<dead>\d+) dead-ball "
                    r"stretches \| graded frames H (?P<H>[0-9.]+)% A (?P<A>[0-9.]+)% D (?P<D>[0-9.]+)%")
GATE = re.compile(r"coverage=(?P<coverage>[0-9.]+) precision=(?P<precision>[0-9.]+) thr=(?P<thr>[0-9.]+)")


def read_steps(path):
    steps = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                steps.append({"step": parts[0], "status": parts[1], "seconds": int(float(parts[2])),
                              "exit_code": int(parts[3]) if parts[3].lstrip("-").isdigit() else parts[3]})
    return steps


def read_summary(path):
    """Scores, per-prediction totals and the video gate from the aggregate-only summary text."""
    out = {"scores": {}, "predictions": {}, "video_gate": [], "reproducible": None, "detections": None}
    current, last_score = None, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            m = re.match(r"\[(OK|FAIL)\]\s+predict (\w+)", line)
            if m:
                current = m.group(2)
            m = SCORE.search(line)
            if m:
                last_score = m["file"]
                out["scores"][last_score] = {k: (float(v) if k in ("spell_f1", "macro_f1") else
                                                 int(v) if k.endswith("possessions") else v)
                                             for k, v in m.groupdict().items() if k != "file"}
            m = BOOT.search(line)
            if m and last_score in out["scores"]:
                out["scores"][last_score]["spell_f1_bootstrap_5_95"] = [float(m["lo"]), float(m["hi"])]
            m = RESULT.search(line)
            if m and current:
                out["predictions"][current] = {"possessions": int(m["possessions"]), "mean_possession_s": float(m["mean_s"]),
                                               "dead_ball_stretches": int(m["dead"]),
                                               "graded_share_pct": {c: float(m[c]) for c in "HAD"}}
            m = GATE.search(line)
            if m:
                out["video_gate"].append({k: float(v) for k, v in m.groupdict().items()})
            if line.startswith("[OK]   identical"):
                out["reproducible"] = True
            elif "DIFFER" in line:
                out["reproducible"] = False
            m = re.search(r"detections saved .*recovered=(\d+) searched=(\d+) share=([0-9.]+)", line)
            if m:
                out["detections"] = {"recovered": int(m[1]), "searched": int(m[2]), "share": float(m[3])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="round")
    ap.add_argument("--steps", required=True, help="tab-separated name, status, seconds, exit code per step")
    ap.add_argument("--summary", required=True, help="the run's aggregate-only summary text")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "benchmarks", "results"))
    args = ap.parse_args(argv)
    plog.setup("run_record")
    try:
        import torch  # noqa: F401 -- only so that environment() can report the GPU
    except Exception:  # noqa: BLE001
        pass
    now = datetime.datetime.now(datetime.timezone.utc)
    steps = read_steps(args.steps)
    record = {"run": "scripts/run_all.sh", "name": args.label, "when_utc": now.isoformat(timespec="seconds"),
              "environment": plog.environment(), "total_seconds": sum(s["seconds"] for s in steps),
              "failed_steps": [s["step"] for s in steps if s["status"] == "failed"], "steps": steps,
              **read_summary(args.summary)}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "{}_run-all.json".format(now.strftime("%Y%m%dT%H%M%SZ")))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
    log.info("benchmark record written", file=plog.rel(out), steps=len(steps), failed=len(record["failed_steps"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
