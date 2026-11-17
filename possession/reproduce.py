"""One command from the data bundle to the submission file (TECHNICAL.md, section 6).

    uv run --no-sync python -m possession.reproduce --name TotalFootball     # Windows or Linux
    make final NAME=TotalFootball                                            # the same, where make exists

Runs the five steps below in order and stops at the first failure, then prints the file's SHA-256:

  1. ball detector self-check on frames where the ATD sees the ball (exit 3, gate not passed, is
     expected: the next step then uses the self-check's best threshold);
  2. ball detection on every graded frame without an ATD ball (about 45 minutes on the RX 9060 XT);
  3. our ATD: the provided one with the detections at least 1 m inside the lines filled in;
  4. the prediction: version-3 rule settings on our ATD;
  5. the official format validator.

--skip-video starts at step 3 and reuses cache/video_ball_fullframe.npz (about a minute).
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

from . import ROOT

# The version-3 rule settings. scripts/run_all.sh uses the same list (tests/test_reproduce.py checks it).
V3_SET = ["--set", "taker_dist=0", "--set", "goal_mouth_half=-1", "--set", "gk_parry_factor=1",
          "--set", "dec_bonus_stop_audio=0", "--set", "emit_shape_weight=0"]
OURS = os.path.join("cache", "atd_ours_inside")
DETECTIONS = os.path.join("cache", "video_ball_fullframe.npz")


def steps(out, skip_video=False):
    """(name, command, extra environment, exit codes that count as success) for each step."""
    py = sys.executable
    plan = []
    if not skip_video:
        plan.append(("detector self-check", [py, "-m", "possession.video.fullframe", "--evaluate"], {}, (0, 3)))
        plan.append(("ball detection", [py, "-m", "possession.video.fullframe", "--force"], {}, (0,)))
    plan.append(("our ATD", [py, "-m", "possession.video.write_atd", "--out-root", OURS, "--inside-m", "1.0"],
                 {}, (0,)))
    plan.append(("prediction", [py, "-m", "possession.run", "--out", out] + V3_SET, {"MTAH_DATA_DIR": OURS}, (0,)))
    plan.append(("format check", [py, os.path.join("evaluation", "validate_submission.py"), out], {}, (0,)))
    return plan


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="submission", help="team name: the file is submissions/<name>.csv")
    ap.add_argument("--out", default=None, help="write here instead of submissions/<name>.csv")
    ap.add_argument("--skip-video", action="store_true", help="reuse the last detections (" + DETECTIONS + ")")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    out = args.out or os.path.join("submissions", args.name + ".csv")

    from games import asset_paths
    missing = [p for p in asset_paths().values() if not os.path.exists(p)]
    if missing:
        print("missing data files (unpack the bundle into data/ with its original names):")
        for p in missing:
            print("   ", os.path.relpath(p))
        return 1
    if args.skip_video and not os.path.isfile(DETECTIONS):
        print("--skip-video needs " + DETECTIONS + "; run once without it")
        return 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    plan = steps(out, args.skip_video)
    t_all = time.time()
    for i, (name, cmd, env, ok_codes) in enumerate(plan, 1):
        print("\n[{}/{}] {}".format(i, len(plan), name), flush=True)
        t0 = time.time()
        code = subprocess.call(cmd, env=dict(os.environ, **env))
        took = round(time.time() - t0)
        if code not in ok_codes:
            print("[FAIL] {} (exit {}, {} s); its log is the newest file in logs/".format(name, code, took))
            return code or 1
        print("[OK]   {} ({} s)".format(name, took), flush=True)

    print("\ndone in {} min: {}".format(round((time.time() - t_all) / 60, 1), out))
    print("sha256 " + sha256(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
