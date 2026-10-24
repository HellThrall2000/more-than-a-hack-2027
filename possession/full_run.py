"""The whole match, end to end, in one command -- with progress on the console and a full log file.

    .venv\\Scripts\\python -m possession.full_run --name <team>                   # everything
    .venv\\Scripts\\python -m possession.full_run --name <team> --skip train,gate,detect,predict_video
    .venv\\Scripts\\python -m possession.full_run --from detect                    # resume at a step
    .venv\\Scripts\\python -m possession.full_run --only audit,baselines

Steps, in order (names for --only / --skip / --from). The four video steps run only with
--with-video: the default submission is the deterministic, tracking-only one.

  check           data bundle present, tactical video has 199,319 frames, GPU visible
  audit           alignment and time-base checks; a mismatch stops the run
  baselines       rebuild the four organiser baselines; they must be byte-identical
  predict_tracking  tracking-only pipeline -> submissions/<name>_tracking.csv, validated, scored
  train           ball detector on the tactical video (GPU); skipped if models/ already has one
  gate            go / no-go: does the detector recover hidden balls? (CPU)
  detect          search the tactical video on every missing-ball frame (CPU, reproducible)
  predict_video   pipeline + video detections -> submissions/<name>_video.csv, validated, scored
  final           submissions/<name>.csv = the video version if the gate passed, else tracking-only
  reports         semantics study (SITUATION.md Q1-Q4) and diagnosis, written to the log

Console: one line per step and progress lines with rate and ETA. logs/<UTC>_full_run.log: the
same plus DEBUG detail, tracebacks and crash dumps. benchmarks/results/<UTC>_full-run.json: step
status, timings and aggregate scores. Public-window scores are a correctness check only: 17
possessions cannot choose between versions (EVALUATION.md).
"""

from __future__ import annotations

import argparse
import datetime
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from . import ROOT
from . import log as plog

log = plog.get(__name__)

STEPS = ("check", "audit", "baselines", "predict_tracking", "train", "gate", "detect", "predict_video",
         "final", "reports")
VIDEO_STEPS = ("train", "gate", "detect", "predict_video")
BASELINES = ("baseline_raw.csv", "baseline_smooth25.csv", "baseline_smooth50.csv", "baseline_smooth125.csv")
N_VIDEO_FRAMES = 199319


class Skip(Exception):
    """Raised inside a step that decides not to run; the reason goes into the summary."""


class StepFailed(Exception):
    pass


class Runner:
    def __init__(self, selected):
        self.selected = selected
        self.rows = []

    def step(self, key, title, fn, critical=True):
        """Run one step under a logged stage; returns its info dict, or None if skipped / failed."""
        if key not in self.selected:
            self.rows.append({"step": key, "status": "not selected", "seconds": 0.0})
            return None
        t0 = time.perf_counter()
        row = {"step": key, "title": title}
        try:
            with plog.stage("[{}/{}] {}".format(STEPS.index(key) + 1, len(STEPS), title)):
                row["info"] = fn() or {}
            row["status"] = "ok"
        except Skip as why:
            row["status"] = "skipped: {}".format(why)
            log.info("    skipped: %s", why)
        except KeyboardInterrupt:
            row["status"] = "interrupted"
            self.rows.append(dict(row, seconds=round(time.perf_counter() - t0, 1)))
            raise
        except Exception as exc:       # noqa: BLE001 -- recorded, then re-raised if critical
            row["status"] = "FAILED: {}: {}".format(type(exc).__name__, exc)
            self.rows.append(dict(row, seconds=round(time.perf_counter() - t0, 1)))
            if critical:
                raise StepFailed(key) from exc
            log.warning("    non-critical step failed; continuing")
            return None
        row["seconds"] = round(time.perf_counter() - t0, 1)
        self.rows.append(row)
        return row.get("info")


def _subprocess(cmd, what, data_dir=None):
    """Run a command, stream its output into the log, raise on a non-zero exit."""
    log.debug("running: %s", " ".join(cmd))
    env = dict(os.environ, MTAH_DATA_DIR=os.path.abspath(data_dir)) if data_dir else None
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip():
            log.info("    | %s", line)
    if proc.returncode:
        raise RuntimeError("{} exited with code {}".format(what, proc.returncode))


def _validate_and_score(path):
    """The organisers' validator on the exact file, then the official metric on the public window."""
    _subprocess([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path],
                "validate_submission.py")
    from .harness import log_result, score_csv
    result = score_csv(path, "public")
    log_result(result, os.path.basename(path), "public")
    info = {k: result.get(k) for k in ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells")}
    log.info("    public window (correctness check only): spellF1 %.4f  macroF1 %.4f  possessions %d (gt %d)",
             info["spell_f1_mean"], info["macro_f1"], info["n_pred_spells"], info["n_gt_spells"])
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--name", default="submission", help="team name: SUBMISSION.md wants <team_name>.csv")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--only", default=None, help="comma list of steps to run")
    ap.add_argument("--skip", default="", help="comma list of steps to leave out")
    ap.add_argument("--from", dest="start", default=None, help="start at this step")
    ap.add_argument("--with-video", action="store_true",
                    help="also run the optional video ball detector (train, gate, detect, predict_video). "
                         "Off by default: its GPU-trained weights are not bit-reproducible and cannot be "
                         "committed yet, so the default submission is the deterministic tracking-only one")
    ap.add_argument("--retrain", action="store_true", help="train the detector even if models/ has one")
    ap.add_argument("--train-device", default="auto")
    ap.add_argument("--detect-device", default="cpu", help="cpu keeps the submission reproducible")
    ap.add_argument("--verbose", action="store_true", help="DEBUG lines on the console too")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Config override passed to every prediction, e.g. --set emit_tag_weight=0")
    ap.add_argument("--summary-dir", default=os.path.join(ROOT, "benchmarks", "results"))
    args = ap.parse_args(argv)
    plog.setup("full_run", verbose=args.verbose)

    for flag, value in (("--only", args.only), ("--skip", args.skip), ("--from", args.start)):
        bad = [s for s in (value or "").split(",") if s and s not in STEPS]
        if bad:
            ap.error("{}: unknown step(s) {}; steps are {}".format(flag, bad, ", ".join(STEPS)))
    selected = list(STEPS[STEPS.index(args.start):] if args.start else STEPS)
    if args.only:
        selected = [s for s in selected if s in args.only.split(",")]
    selected = [s for s in selected if s not in args.skip.split(",")]
    if not args.with_video and not args.only:
        selected = [s for s in selected if s not in VIDEO_STEPS]
    log.info("full run '%s': steps %s%s", args.name, ", ".join(selected),
             "" if args.with_video else "  (video detector off; --with-video to add it)")

    data = ["--data-dir", args.data_dir] if args.data_dir else []
    data_set = data + [x for item in args.set for x in ("--set", item)]
    sub_dir = os.path.join(ROOT, "submissions")
    tracking_csv = os.path.join(sub_dir, "{}_tracking.csv".format(args.name))
    video_csv = os.path.join(sub_dir, "{}_video.csv".format(args.name))
    final_csv = os.path.join(sub_dir, "{}.csv".format(args.name))
    model = os.path.join(ROOT, "models", "ball_detector.pt")
    video_ball = os.path.join(ROOT, "cache", "video_ball.npz")
    state = {"gate_passed": None}
    run = Runner(selected)

    def gate_passed():
        """This run's gate result; else a saved one, if it is newer than the detector it judged."""
        if state["gate_passed"] is not None:
            return state["gate_passed"]
        from .video.detect import GATE_FILE
        if os.path.isfile(GATE_FILE) and os.path.isfile(model) \
                and os.path.getmtime(GATE_FILE) > os.path.getmtime(model):
            with open(GATE_FILE) as handle:
                g = json.load(handle)["gate"]
            state["gate_passed"] = bool(g["met"] and g["worthwhile"])
            log.info("    using the saved gate result in %s: %s", plog.rel(GATE_FILE),
                     "passed" if state["gate_passed"] else "not passed")
            return state["gate_passed"]
        return False
    t0 = time.perf_counter()

    def check():
        from games import asset_paths
        paths = asset_paths(data_dir=args.data_dir, source="tactical")
        missing = [k for k, p in paths.items() if not os.path.isfile(p)]
        for key, path in paths.items():
            log.info("    %-14s %s %s", key, "ok     " if key not in missing else "MISSING", plog.rel(path))
        if missing:
            raise FileNotFoundError("tactical assets missing: {} (unpack the bundle into data/, DATA.md)".format(missing))
        info = {}
        try:
            import cv2
            cap = cv2.VideoCapture(paths["video"])
            info["video_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            log.info("    tactical video reports %d frames (expected %d)", info["video_frames"], N_VIDEO_FRAMES)
            if info["video_frames"] != N_VIDEO_FRAMES:
                log.warning("    frame count differs: this is not the reference decode (FAQ.md) -- video steps "
                            "may misalign")
        except ImportError:
            log.warning("    OpenCV not installed: the video steps will fail (pip install -r requirements-video.txt)")
        if any(s in selected for s in ("train", "gate", "detect")):
            try:
                import torch
                from .video.model import pick_device
                dev = pick_device(args.train_device)
                info["train_device"] = str(dev)
                log.info("    torch %s, training device %s%s", torch.__version__, dev,
                         " ({})".format(torch.cuda.get_device_name(0)) if dev.type == "cuda" else "")
            except ImportError:
                log.warning("    torch not installed: the video steps will fail")
        return info

    try:
        run.step("check", "check the data bundle and the environment", check)

        def audit():
            from .audit import main as audit_main
            if audit_main(data) != 0:
                raise RuntimeError("alignment mismatch -- stop and report it (see the audit lines above)")
        run.step("audit", "audit: alignment and data statistics", audit)

        def baselines():
            out = os.path.join(ROOT, "cache", "baselines")
            os.makedirs(out, exist_ok=True)
            _subprocess([sys.executable, os.path.join(ROOT, "starter", "baseline.py"), "--out", out],
                        "starter/baseline.py", args.data_dir)
            same = {b: filecmp.cmp(os.path.join(out, b), os.path.join(ROOT, "evaluation", "results", b), shallow=False)
                    for b in BASELINES}
            for b, ok in same.items():
                log.info("    %-24s %s", b, "identical" if ok else "DIFFERENT")
            if not all(same.values()):
                raise RuntimeError("baselines differ from the committed ones: the environment does not "
                                   "reproduce the organisers' files")
            return {"identical": True}
        run.step("baselines", "rebuild the organiser baselines (byte-identical check)", baselines, critical=False)

        def predict_tracking():
            from .run import main as run_main
            if run_main(["--out", tracking_csv] + data_set) != 0:
                raise RuntimeError("prediction produced an invalid file")
            return _validate_and_score(tracking_csv)
        run.step("predict_tracking", "tracking-only prediction over the whole match", predict_tracking)

        def train():
            if os.path.isfile(model) and not args.retrain:
                raise Skip("{} exists (pass --retrain to train again)".format(plog.rel(model)))
            from .video.train import main as train_main
            train_main(["--device", args.train_device] + data)
            return {"model": plog.rel(model)}
        run.step("train", "train the ball detector on the tactical video", train, critical=False)

        def gate():
            from .video.detect import GATE_FILE, main as detect_main
            if not os.path.isfile(model):
                raise Skip("no detector at {}".format(plog.rel(model)))
            if os.path.isfile(GATE_FILE):
                os.remove(GATE_FILE)            # never let an old result stand in for this one
            rc = detect_main(["--evaluate", "--device", args.detect_device, "--model", model] + data)
            if rc not in (0, 3):
                raise RuntimeError("the gate exited with code {}".format(rc))
            state["gate_passed"] = rc == 0
            with open(GATE_FILE) as handle:
                result = json.load(handle)
            return {"passed": state["gate_passed"], "overall": result["overall"], "gate": result["gate"]}
        run.step("gate", "go / no-go gate for the video detections", gate, critical=False)

        def detect():
            if not os.path.isfile(model):
                raise Skip("no detector at {}".format(plog.rel(model)))
            if not gate_passed():
                raise Skip("the gate has not passed for the current detector")
            from .video.detect import main as detect_main
            if detect_main(["--device", args.detect_device, "--model", model, "--out", video_ball] + data) != 0:
                raise RuntimeError("detection failed")
            return {"detections": plog.rel(video_ball)}
        run.step("detect", "search the tactical video on every missing-ball frame", detect, critical=False)

        def predict_video():
            if not gate_passed():
                raise Skip("the gate has not passed for the current detector")
            if not os.path.isfile(video_ball):
                raise Skip("no detections at {}".format(plog.rel(video_ball)))
            from .run import main as run_main
            if run_main(["--out", video_csv, "--video-ball", video_ball] + data_set) != 0:
                raise RuntimeError("prediction produced an invalid file")
            return _validate_and_score(video_csv)
        run.step("predict_video", "prediction with the video detections", predict_video, critical=False)

        def final():
            use_video = args.with_video and os.path.isfile(video_csv) and gate_passed() and \
                any(r["step"] == "predict_video" and r["status"] == "ok" for r in run.rows)
            source = video_csv if use_video else tracking_csv
            if not os.path.isfile(source):
                raise RuntimeError("no prediction to promote: run predict_tracking first")
            shutil.copyfile(source, final_csv)
            log.info("    %s <- %s (%s)", plog.rel(final_csv), os.path.basename(source),
                     "video detections, gate passed" if use_video else "tracking only")
            _subprocess([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), final_csv],
                        "validate_submission.py")
            with open(final_csv, "rb") as handle:
                sha = hashlib.sha256(handle.read()).hexdigest()
            log.info("    SHA-256 %s  -- a rerun must print the same (RULES.md: the same file)", sha)
            return {"final": plog.rel(final_csv), "from": os.path.basename(source), "sha256": sha}
        run.step("final", "choose and validate the submission file", final)

        def reports():
            from .diagnose import main as diagnose_main
            from .semantics import main as semantics_main
            semantics_main(data_set)
            diagnose_main(data_set)
        run.step("reports", "semantics study and diagnosis (aggregates, see the log)", reports, critical=False)
        failed = None
    except StepFailed as exc:
        failed = str(exc)
        log.error("stopping: step '%s' failed -- the traceback is in %s", failed, plog.rel(plog.log_path()))
        log.debug("cause", exc_info=exc.__cause__)
    except KeyboardInterrupt:
        failed = "interrupted"
        log.warning("interrupted -- resume with:  python -m possession.full_run --from <step>")

    total = time.perf_counter() - t0
    now = datetime.datetime.now(datetime.timezone.utc)
    summary = {"run": "possession.full_run", "name": args.name, "when_utc": now.isoformat(timespec="seconds"),
               "video": "tactical", "seconds": round(total, 1), "failed_step": failed,
               "log_file": plog.rel(plog.log_path()), "environment": plog.environment(),
               "steps": run.rows}
    os.makedirs(args.summary_dir, exist_ok=True)
    out = os.path.join(args.summary_dir, "{}_full-run.json".format(now.strftime("%Y%m%dT%H%M%SZ")))
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    log.info("")
    log.info("%-18s %-9s %s", "step", "time", "status")
    for row in run.rows:
        log.info("%-18s %-9s %s", row["step"], plog.duration(row.get("seconds", 0.0)), row["status"])
    log.info("total %s | summary %s | log %s", plog.duration(total), plog.rel(out),
             plog.rel(plog.log_path()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
